import os.path
import wandb
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, BertConfig, BertForSequenceClassification
from torch.optim import Adam
import numpy as np
import random
from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader, get_eval_loader
from wilds.datasets.wilds_dataset import WILDSSubset
import transforms_
from wilds_exps_utils.wilds_configs import datasets as dataset_configs
from types import SimpleNamespace
import torch.nn.functional as F
from gdro_fork.data.confounder_utils import *
from gdro_fork.loss import LossComputer

from utils import *
from settings import *
import torchvision
from typing import List, Optional, Tuple, Union
from transformers.modeling_outputs import  SequenceClassifierOutput

from torch.utils.data import Dataset, DataLoader


class TextDataset(Dataset):
    def __init__(self, file_path, transform=None, split=False):
        data_df = pd.read_csv(file_path, encoding='latin1')

        if split == True:
            ind = np.arange(len(data_df))
            rng = np.random.default_rng(0)
            rng.shuffle(ind)
            n_train = int(0.8 * len(ind))
            ind1 = ind[:n_train]
            ind2 = ind[n_train:]
            self.data = data_df[ind1]
        else:
            self.data = data_df

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        label = row['label']
        text = row['text']
        group_id = row['group_ids']

        label = torch.tensor(label, dtype=torch.long)

        if self.transform:
            text = self.transform(text)

        return text, label, group_id



def load_data(file_path, batch_size=32, transform=None, shuffle=True):
    dataset = TextDataset(file_path, transform)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return dataloader


class BertForSequenceClassificationWithCovReg(BertForSequenceClassification):
    def __init__(self, config, feature_size, device, reg_disentangle, reg_causal=0, disentangle_en=False,
                 counterfactual_en=False):
        super().__init__(config)
        self.feature_size = feature_size

        self.custom_linear = nn.Linear(config.hidden_size, self.feature_size)
        self.classifier = nn.Linear(self.feature_size, config.num_labels)
        self.mask = ~torch.eye(self.feature_size, dtype=bool).to(device)

        self.reg_disentangle = reg_disentangle
        self.disentangle_en = disentangle_en

        self.reg_causal = reg_causal
        self.counterfactual_en = counterfactual_en

        # Initialize weights and apply final processing
        self.post_init()


    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        weights=None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        features = self.custom_linear(pooled_output)
        logits = self.classifier(features)

        loss = None

        if labels is not None:
            # For robust learning
            if self.disentangle_en == True:
                covariance = self.compute_covariance(features)
                regularization = torch.norm(covariance - torch.diag_embed(torch.diagonal(covariance)), p='fro')
            else:
                regularization = 0

            if self.counterfactual_en == True:
                causal_regularization = self.counterfact(features, labels, logits)
            else:
                causal_regularization = None
            # End

            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                # Modified
                loss_fct = nn.CrossEntropyLoss(reduction='none')
                print("Problem type is single_label_classification.")
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                if causal_regularization is not None:
                    loss += self.reg_causal * causal_regularization
                if weights is not None:
                    loss *= weights
                loss = loss.mean() + self.reg_disentangle * regularization
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def compute_covariance(self, features):
        feature_mean = torch.mean(features, dim=0, keepdim=True)
        features = features - feature_mean
        covariance_matrix = features.T @ features / (features.size(0) - 1)
        return covariance_matrix


class BertClassifierWithCovReg(nn.Module):
    def __init__(self, model_name, num_labels, feature_size, device, reg, reg_causal=0, disentangle_en=False,
                 counterfactual_en=False):
        super().__init__()
        self.device = device
        self.num_labels = num_labels
        self.feature_size = feature_size

        self.bert = BertModel.from_pretrained(model_name)
        self.linear = nn.Linear(self.bert.config.hidden_size, self.feature_size)
        self.activation = nn.Tanh()
        # self.feature_size = self.bert.config.hidden_size
        self.classifier = nn.Linear(self.feature_size, num_labels, bias=True)
        self.crossEntropyLoss = nn.CrossEntropyLoss(reduction='none')


        self.mask = ~torch.eye(self.feature_size, dtype=bool).to(device)

        self.reg = reg
        self.disentangle_en = disentangle_en

        self.reg_causal = reg_causal
        self.counterfactual_en = counterfactual_en

    def forward(self, input_ids, attention_mask, labels=None, weights=None, token_type_ids=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs.pooler_output
        # feature = pooled_output
        feature = self.linear(pooled_output)
        feature = self.activation(feature)
        logits = self.classifier(feature)

        total_loss = 0
        causal_regularization = None
        if labels is not None:

            if self.disentangle_en == True:
                covariance = self.compute_covariance(feature)
                regularization = torch.norm(covariance - torch.diag_embed(torch.diagonal(covariance)), p='fro')
                total_loss += self.reg * regularization

            if self.counterfactual_en == True:
                causal_regularization = self.counterfact(feature, labels, logits)
                # total_loss += self.reg_causal * causal_regularization

            reg = self.classifier.weight.pow(2).sum() + self.classifier.bias.pow(2).sum()
            total_loss += 2 * reg
            total_loss += self.get_total_loss(logits, labels, weights, causal_regularization)

        return logits, total_loss

    def get_total_loss(self, logits, labels, weights, causal_regularization):
        total_loss = self.crossEntropyLoss(logits.view(-1, self.num_labels), labels.view(-1))
        if causal_regularization is not None:
            total_loss += self.reg_causal * causal_regularization
        if weights is not None:
            total_loss *= weights

        # total_loss = total_loss.mean()
        total_loss = total_loss.sum()

        return total_loss

    # def get_loss(self, logits, labels, weights, causal_regularization):
    #     total_loss = self.crossEntropyLoss(logits.view(-1, self.num_labels), labels.view(-1))
    #     if weights is not None:
    #         total_loss *= weights
    #         total_loss = total_loss.sum()
    #     else:
    #         total_loss = total_loss.mean()
    #
    #     if causal_regularization is not None:
    #         total_loss += self.reg_causal * causal_regularization.mean()
    #
    #     return total_loss

    def compute_covariance(self, features):
        feature_mean = torch.mean(features, dim=0, keepdim=True)
        features = features - feature_mean
        covariance_matrix = features.T @ features / (features.size(0) - 1)
        return covariance_matrix

    def counterfact(self, feature, labels, logits):
        labels = labels.clone().detach()

        ind = F.one_hot(labels, self.num_labels)
        prob_raw = torch.sum(F.softmax(logits, dim=-1) * ind, 1).clone().detach()
        prob_raw = prob_raw.repeat_interleave(self.feature_size).view(-1)

        feature = feature.view(-1, 1, self.feature_size)
        feature = feature * self.mask
        feature = feature.view(-1, self.feature_size)
        logits_counterfactual = self.classifier(feature)
        labels = labels.repeat_interleave(self.feature_size).view(-1)
        prob_sub = F.softmax(logits_counterfactual, dim=-1)[torch.arange(labels.shape[0]), labels]

        z = prob_raw - prob_sub + 1
        z = torch.where(z > 1, z, torch.tensor(1.0).to(self.device)).view(-1, self.feature_size)
        log_cpns = torch.mean(torch.log(z), dim=-1)
        causal_constraints = -log_cpns

        return causal_constraints


class Bert(nn.Module):
    def __init__(self, model_name, num_labels, feature_size, device, reg, reg_causal=0, disentangle_en=False,
                 counterfactual_en=False, hidden_dropout_prob=0.1):
        super().__init__()
        self.device = device
        self.num_labels = num_labels

        self.bert = BertModel.from_pretrained(model_name)
        self.feature_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.classifier = nn.Linear(self.feature_size, num_labels)
        self.crossEntropyLoss = nn.CrossEntropyLoss(reduction='none')

        self.mask = ~torch.eye(self.feature_size, dtype=bool).to(device)

        self.reg = reg
        self.disentangle_en = disentangle_en

        self.reg_causal = reg_causal
        self.counterfactual_en = counterfactual_en

    def forward(self, input_ids, attention_mask, labels=None, weights=None, token_type_ids=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        feature = outputs.pooler_output
        feature = self.dropout(feature)
        logits = self.classifier(feature)

        total_loss = 0
        causal_regularization = None
        if labels is not None:

            if self.disentangle_en == True:
                covariance = self.compute_covariance(feature)
                regularization = torch.norm(covariance - torch.diag_embed(torch.diagonal(covariance)), p='fro')
                total_loss += self.reg * regularization

            if self.counterfactual_en == True:
                causal_regularization = self.counterfact(feature, labels, logits)
                # total_loss += self.reg_causal * causal_regularization

            total_loss += self.get_total_loss(logits, labels, weights, causal_regularization)

        return logits, total_loss

    def get_total_loss(self, logits, labels, weights, causal_regularization):
        total_loss = self.crossEntropyLoss(logits.view(-1, self.num_labels), labels.view(-1))
        if causal_regularization is not None:
            total_loss += self.reg_causal * causal_regularization
        if weights is not None:
            total_loss *= weights

        total_loss = total_loss.mean()

        return total_loss

    # def get_loss(self, logits, labels, weights, causal_regularization):
    #     total_loss = self.crossEntropyLoss(logits.view(-1, self.num_labels), labels.view(-1))
    #     if weights is not None:
    #         total_loss *= weights
    #         total_loss = total_loss.sum()
    #     else:
    #         total_loss = total_loss.mean()
    #
    #     if causal_regularization is not None:
    #         total_loss += self.reg_causal * causal_regularization.mean()
    #
    #     return total_loss

    def compute_covariance(self, features):
        feature_mean = torch.mean(features, dim=0, keepdim=True)
        features = features - feature_mean
        covariance_matrix = features.T @ features / (features.size(0) - 1)
        return covariance_matrix

    def counterfact(self, feature, labels, logits):
        labels = labels.clone().detach()

        ind = F.one_hot(labels, self.num_labels)
        prob_raw = torch.sum(F.softmax(logits, dim=-1) * ind, 1).clone().detach()
        prob_raw = prob_raw.repeat_interleave(self.feature_size).view(-1)

        feature = feature.view(-1, 1, self.feature_size)
        feature = feature * self.mask
        feature = feature.view(-1, self.feature_size)
        logits_counterfactual = self.classifier(feature)
        labels = labels.repeat_interleave(self.feature_size).view(-1)
        prob_sub = F.softmax(logits_counterfactual, dim=-1)[torch.arange(labels.shape[0]), labels]

        z = prob_raw - prob_sub + 1
        z = torch.where(z > 1, z, torch.tensor(1.0).to(self.device)).view(-1, self.feature_size)
        log_cpns = torch.mean(torch.log(z), dim=-1)
        causal_constraints = -log_cpns

        return causal_constraints


def get_civil_data(task_config, dataset):
    full_dataset = get_dataset(dataset=dataset, download=False, root_dir=task_config.root_dir)
    transform = transforms_.initialize_transform(
        transform_name=task_config.transform,
        config=task_config,
        dataset=full_dataset,
        additional_transform_name=None,
        is_training=False)

    test_data = full_dataset.get_subset("test", transform=transform)
    val_data = full_dataset.get_subset("val", transform=transform)
    train_data = full_dataset.get_subset("train", transform=transform)

    if task_config.dfr_reweighting_drop:
        idx = train_data.indices.copy()
        rng = np.random.default_rng(task_config.dfr_reweighting_seed)
        rng.shuffle(idx)
        n_train = int((1 - task_config.dfr_reweighting_frac) * len(idx))
        train_idx = idx[:n_train]
        val_idx = idx[n_train:]
        train_data = WILDSSubset(
            full_dataset,
            indices=train_idx,
            transform=transform
        )
        reweighting_data = WILDSSubset(
            full_dataset,
            indices=val_idx,
            transform=transform
        )
    else:
        reweighting_data = val_data
    return train_data, val_data, test_data, reweighting_data


def get_mnli_data(args, train=False, return_full_dataset=False):
    train_data, val_data, test_data = prepare_confounder_data(args, train, return_full_dataset)
    if args.dfr_reweighting_drop:
        print(f'Dropping DFR reweighting data, seed {args.dfr_reweighting_seed}')

        idx = train_data.dataset.indices.copy()
        rng = np.random.default_rng(args.dfr_reweighting_seed)
        rng.shuffle(idx)
        n_train = int((1 - args.dfr_reweighting_frac) * len(idx))
        train_idx = idx[:n_train]

        print(f'Original dataset size: {len(train_data.dataset.indices)}')
        train_data.dataset = torch.utils.data.dataset.Subset(
            train_data.dataset.dataset,
            indices=train_idx)
        print(f'New dataset size: {len(train_data.dataset.indices)}')

    return train_data, val_data, test_data


def get_data(task_config, dataset, train=False, return_full_dataset=False):
    if dataset == 'civilcomments':
        return get_civil_data(task_config, dataset)

    elif dataset == 'MultiNLI':
        return prepare_confounder_data(task_config, train, return_full_dataset)


def compute_accuracy(logits, labels):
    predictions = torch.argmax(logits, dim=-1)
    correct = (predictions == labels).float()
    accuracy = correct.sum() / len(correct)
    return accuracy


def compute_group_avg(losses, group_idx, n_groups):
    group_map = (group_idx == torch.arange(n_groups).unsqueeze(1).long()).float()
    group_count = group_map.sum(1)
    group_denom = group_count + (group_count == 0).float()  # avoid nans
    group_loss = (group_map @ losses.view(-1)) / group_denom
    return group_loss, group_count


def model_parameters_freeze(model):
    # Freeze layers of the pretrained model except the last linear layer
    for name, param in model.named_parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        # nn.init.normal_(param, mean=0, std=1)
        param.requires_grad = True

    print("\nAfter fixing the layers before the last linear layer:")
    for name, param in model.named_parameters():
        print(name, param.requires_grad)

    return model


def compute_weights(logits, labels, gamma, balance_classes, all_group_ids=None):
    # AFR
    with torch.no_grad():
        p = logits.softmax(-1)
        y_onehot = torch.zeros_like(logits).scatter_(-1, labels.unsqueeze(-1), 1)
        p_true = (p * y_onehot).sum(-1)
        weights = (-gamma * p_true).exp()
        n_classes = torch.unique(labels).numel()
        if balance_classes:
            if n_classes == 2:
                w1 = (labels == 0).sum()
                w2 = (labels == 1).sum()
                weights[labels == 0] *= w2 / w1
            else:
                class_count = []
                for y in range(n_classes):
                    class_count.append((labels == y).sum())
                for y in range(1, n_classes):
                    weights[labels == y] *= class_count[0] / class_count[y]
        weights = weights.detach()
        weights /= weights.sum()

    return weights

def compute_weights1(logits, labels, gamma, balance_classes, all_group_ids=None):
    with torch.no_grad():
        predictions = torch.argmax(logits, dim=1)
        correct_predictions = predictions == labels
        n_classes = torch.unique(labels).numel()
        correct_counts = torch.zeros(n_classes, dtype=torch.int32)
        incorrect_counts = torch.zeros(n_classes, dtype=torch.int32)
        total_counts = len(labels)
        weights = torch.ones_like(labels, dtype=torch.float32)

        for i in range(n_classes):
            correct_counts[i] = (correct_predictions & (labels == i)).sum()
            incorrect_counts[i] = (~correct_predictions & (labels == i)).sum()
        for i in range(n_classes):
            print(f"Class {i}: Correct: {correct_counts[i]}, Incorrect: {incorrect_counts[i]}")

        correct_ratios = correct_counts.float() / total_counts
        incorrect_ratios = incorrect_counts.float() / total_counts
        correct_weights = torch.where(correct_counts > 0, 1.0 / correct_ratios, torch.tensor(1.0))
        incorrect_weights = torch.where(incorrect_counts > 0, 1.0 / incorrect_ratios, torch.tensor(1.0))

        for i in range(n_classes):
            weights[(labels == i) & correct_predictions] = correct_weights[i]
            weights[(labels == i) & (~correct_predictions)] = incorrect_weights[i]


    return weights


def compute_weights2(logits, labels, gamma, balance_classes, all_group_ids=None):
    with torch.no_grad():
        predictions = torch.argmax(logits, dim=1)
        correct_predictions = predictions == labels
        n_classes = torch.unique(labels).numel()
        correct_counts = torch.zeros(n_classes, dtype=torch.int32)
        incorrect_counts = torch.zeros(n_classes, dtype=torch.int32)
        total_counts = len(labels)
        weights = torch.ones_like(labels, dtype=torch.float32)

        for i in range(n_classes):
            correct_counts[i] = (correct_predictions & (labels == i)).sum()
            incorrect_counts[i] = (~correct_predictions & (labels == i)).sum()
        for i in range(n_classes):
            print(f"Class {i}: Correct: {correct_counts[i]}, Incorrect: {incorrect_counts[i]}")

        correct_ratios = correct_counts.float() / (correct_counts + incorrect_counts)
        incorrect_ratios = incorrect_counts.float() / (correct_counts + incorrect_counts)
        correct_weights = torch.where(correct_counts > 0, 1.0 / correct_ratios, torch.tensor(1.0))
        incorrect_weights = torch.where(incorrect_counts > 0, 1.0 / incorrect_ratios, torch.tensor(1.0))

        for i in range(n_classes):
            weights[(labels == i) & correct_predictions] = correct_weights[i]
            weights[(labels == i) & (~correct_predictions)] = incorrect_weights[i]

    return weights


def compute_weights3(logits, labels, gamma, balance_classes, all_group_ids):
    with torch.no_grad():
        group_ids, group_sizes = torch.unique(all_group_ids, return_counts=True)
        total_counts = len(labels)
        group_ratios = group_sizes/total_counts
        weights = torch.ones_like(labels, dtype=torch.float32)

        for i, group_id in enumerate(group_ids):
            weights[(all_group_ids == group_id)] = 1.0 / group_ratios[i]

    return weights

def compute_weights4(logits, labels, gamma, balance_classes=None, all_group_ids=None):
    # JTT
    with torch.no_grad():
        predictions = torch.argmax(logits, dim=1)
        correct_predictions = predictions == labels
        n_classes = torch.unique(labels).numel()

        weights = torch.ones_like(labels, dtype=torch.float32)

        for i in range(n_classes):
            weights[(labels == i) & correct_predictions] = 1
            weights[(labels == i) & (~correct_predictions)] = gamma

    return weights


def evaluation(model, dataset, dataloader, device):
    model.eval()
    with torch.no_grad():
        all_predictions, all_y_true, all_metadata, all_logits = [], [], [], []
        for batch in dataloader:
            input_ids = batch[0][:, :, 0].to(device)
            attention_mask = batch[0][:, :, 1].to(device)
            labels = batch[1].to(device)
            metadata = batch[2]

            logits, test_loss = model(input_ids, attention_mask, labels)
            predictions = torch.argmax(logits, axis=1)

            all_logits.append(logits.cpu())
            all_predictions.append(predictions.cpu())
            all_y_true.append(labels.cpu())
            all_metadata.append(metadata.cpu())

        all_logits = torch.cat(all_logits, axis=0)
        all_predictions = torch.cat(all_predictions, axis=0)
        all_y_true = torch.cat(all_y_true, axis=0)
        all_metadata = torch.cat(all_metadata, axis=0)

        total_loss = model.crossEntropyLoss(all_logits.view(-1, model.num_labels), all_y_true.view(-1)).mean()
        total_accuracy = compute_accuracy(all_logits, all_y_true)
        results = dataset.eval(all_predictions.cpu(), all_y_true.cpu(), all_metadata.cpu())

    return results, total_loss, total_accuracy


def evaluation_nli(model, dataset_n_groups, dataloader, device):
    model.eval()
    with torch.no_grad():
        all_predictions, all_y_true, all_group_idx, all_losses, all_logits = [], [], [], [], []
        for batch in dataloader:
            input_ids = batch[0][:, :, 0].to(device)
            attention_mask = batch[0][:, :, 1].to(device)
            labels = batch[1].to(device)
            group_ids = batch[2]

            logits, test_loss = model(input_ids, attention_mask, labels)
            loss = model.crossEntropyLoss(logits.view(-1, model.num_labels), labels.view(-1))
            predictions = torch.argmax(logits, axis=1)

            all_logits.append(logits.cpu())
            all_predictions.append(predictions.cpu())
            all_y_true.append(labels.cpu())
            all_group_idx.append(group_ids.cpu())
            all_losses.append(loss.cpu())

        all_logits = torch.cat(all_logits, axis=0)
        all_predictions = torch.cat(all_predictions, axis=0)
        all_y_true = torch.cat(all_y_true, axis=0)
        all_group_idx = torch.cat(all_group_idx, axis=0)
        all_losses = torch.cat(all_losses, axis=0)

        total_loss = all_losses.mean()
        total_accuracy = compute_accuracy(all_logits, all_y_true)

        group_acc, _ = compute_group_avg((all_predictions == all_y_true).float(), all_group_idx, dataset_n_groups)
        group_losses, _ = compute_group_avg(all_losses, all_group_idx, dataset_n_groups)

    return group_acc, total_loss, total_accuracy, group_losses


def get_data_loader(dataset_name, data, task_config, train=True, loader_type="standard", **loader_kwargs):
    data_loader = None
    if dataset_name == 'civilcomments':
        if train == True:
            data_loader = get_train_loader(loader_type, data, batch_size=task_config.batch_size,
                                           uniform_over_groups=False)
        else:
            data_loader = get_eval_loader(loader_type, data, batch_size=task_config.batch_size)

    elif dataset_name == 'MultiNLI':
        if data is not None:
            if train == True:
                data_loader = data.get_loader(train=train, reweight_groups=task_config.reweight_groups, **loader_kwargs)
            else:
                data_loader = data.get_loader(train=train, reweight_groups=None, **loader_kwargs)

    return data_loader


class ResNet50withCovReg(nn.Module):
    def __init__(self, reduce_dim, output_dim, device, reg=0, reg_causal=0, disentangle_en=False,
                 counterfactual_en=False):
        super(ResNet50withCovReg, self).__init__()
        self.device = device
        self.num_labels = output_dim
        self.feature_size = reduce_dim

        self.resnet50 = torchvision.models.resnet50(pretrained=True)
        self.resnet50.fc = nn.Linear(self.resnet50.fc.in_features, reduce_dim)
        self.activation = nn.Tanh()
        self.classifier = nn.Linear(reduce_dim, output_dim)
        self.crossEntropyLoss = nn.CrossEntropyLoss(reduction='none')

        self.mask = ~torch.eye(self.feature_size, dtype=bool).to(device)

        self.reg = reg
        self.disentangle_en = disentangle_en

        self.reg_causal = reg_causal
        self.counterfactual_en = counterfactual_en

    def forward(self, x, labels=None, weights=None):
        feature = self.resnet50(x)
        feature = self.activation(feature)
        logits = self.classifier(feature)

        total_loss = 0
        causal_regularization = None
        if labels is not None:

            if self.disentangle_en == True:
                regularization = self.compute_disentangle_loss(feature)
                total_loss += self.reg * regularization

            if self.counterfactual_en == True:
                causal_regularization = self.counterfact(feature, labels, logits)
                # total_loss += self.reg_causal * causal_regularization

            total_loss += self.get_total_loss(logits, labels, weights, causal_regularization)

        return logits, total_loss

    def get_total_loss(self, logits, labels, weights, causal_regularization):
        total_loss = self.crossEntropyLoss(logits.view(-1, self.num_labels), labels.view(-1))
        if causal_regularization is not None:
            total_loss += self.reg_causal * causal_regularization
        if weights is not None:
            total_loss *= weights

        total_loss = total_loss.mean()

        return total_loss

    def compute_disentangle_loss(self, features):
        feature_mean = torch.mean(features, dim=0, keepdim=True)
        centered_features = features - feature_mean
        covariance_matrix = centered_features.T @ centered_features / (centered_features.size(0) - 1)
        disentangle_loss = torch.norm(covariance_matrix - torch.diag_embed(torch.diagonal(covariance_matrix)), p='fro')
        return disentangle_loss

    def counterfact(self, feature, labels, logits):
        labels = labels.clone().detach()

        ind = F.one_hot(labels, self.num_labels)
        prob_raw = torch.sum(F.softmax(logits, dim=-1) * ind, 1).clone().detach()
        prob_raw = prob_raw.repeat_interleave(self.feature_size).view(-1)

        feature = feature.view(-1, 1, self.feature_size)
        feature = feature * self.mask
        feature = feature.view(-1, self.feature_size)
        logits_counterfactual = self.classifier(feature)
        labels = labels.repeat_interleave(self.feature_size).view(-1)
        prob_sub = F.softmax(logits_counterfactual, dim=-1)[torch.arange(labels.shape[0]), labels]

        z = prob_raw - prob_sub + 1
        z = torch.where(z > 1, z, torch.tensor(1.0).to(self.device)).view(-1, self.feature_size)
        log_cpns = torch.mean(torch.log(z), dim=-1)
        causal_constraints = -log_cpns

        return causal_constraints

# def main():
#     disentangle_version = 3
#     cpns_version = 1
#     reweight_version = 1
#
#     DATASET = 'civilcomments'
#     reg = 0.5
#     lr = 2e-5
#     n_epochs = 10
#     batch_size = 32
#     model_name = 'bert-base-uncased'
#     dfr_reweighting_frac = 0 # 0.2
#
#     num_classes = 2
#     feature_size = 100
#     total_weights = None
#     finetune_flg = True # False
#     reweight_flg = True
#     weight_decay = 0
#
#     seed = 42
#     root_dir = '../../data/'
#     data_dir = root_dir + 'datasets/'
#     model_save_path = root_dir + 'models/'
#     if not os.path.exists(model_save_path):
#         os.makedirs(model_save_path)
#
#     best_model = None
#     best_loss = float('inf')
#     best_acc_wg = 0
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print("Using device:", device)
#
#     if finetune_flg == True:
#         load_model_path = model_save_path + f'best_model_{disentangle_version}.pth'
#         best_model_path = model_save_path + f'best_model_{disentangle_version}_{cpns_version}_{reweight_version}.pth'
#         load_local_model = True
#         reg_causal = 0.1
#         disentangle_en = False
#         counterfactual_en = True
#     else:
#         best_model_path = model_save_path + f'best_model_{disentangle_version}.pth'
#         load_local_model = False
#         reg_causal = 0
#         disentangle_en = True
#         counterfactual_en = False
#
#     if reweight_flg == True:
#         gamma = 0.5
#     else:
#         gamma = 0
#
#
#     wandb.init(
#         # set the wandb project where this run will be logged
#         project="robust-learning",
#         name=f"disentangle_cov_reg_{disentangle_version}_{cpns_version}_{reweight_version}",
#         # notes="Bert with the regularization of the covariance matrix of the input of the last layer",
#         # notes="Finetune the disentangled Bert, with the causality constraints, initialize the last later",
#         notes="Finetune the disentangled Bert, with the causality constraints, initialize the last later, reweights the CE loss",
#         # track hyperparameters and run metadata
#         config={
#             "learning_rate": lr,
#             "architecture": model_name,
#             "dataset": DATASET,
#             "epochs": n_epochs,
#             "batch_size": batch_size,
#             "dfr_reweighting_frac": dfr_reweighting_frac,
#             "Regularization coefficient": reg,
#             "Bert feature size": feature_size,
#             "Causal Regularization coefficient": reg_causal,
#             "gamma": gamma,
#             "seed": seed,
#             "weight_decay": weight_decay,
#         }
#     )
#     wandb.define_metric("epoch")
#     wandb.define_metric("Train Loss", step_metric='epoch')
#     wandb.define_metric("Train Accuracy", step_metric='epoch')
#     wandb.define_metric("Validation Loss", step_metric='epoch')
#     wandb.define_metric("Validation Accuracy", step_metric='epoch')
#     wandb.define_metric("Best Validation Loss", step_metric='epoch')
#     wandb.define_metric("Best Validation Accuracy", step_metric='epoch')
#     wandb.define_metric("Best Validation Worst Group Accuracy", step_metric='epoch')
#
#     set_seed(seed)
#     model = BertClassifierWithCovReg(model_name, num_labels=num_classes, feature_size=feature_size, device=device, reg=reg, reg_causal=reg_causal, disentangle_en=disentangle_en, counterfactual_en=counterfactual_en).to(device)
#     if load_local_model:
#         model.load_state_dict(torch.load(load_model_path, map_location=device))
#         model = model_parameters_freeze(model)
#     # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#     optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
#
#     args = SimpleNamespace(
#         root_dir=data_dir,
#         batch_size=batch_size,
#         dfr_reweighting_drop=True,
#         dfr_reweighting_seed=seed,
#         dfr_reweighting_frac=dfr_reweighting_frac,
#     )
#     wilds_config = SimpleNamespace(
#         algorithm='ERM',
#         load_featurizer_only=False,
#         pretrained_model_path=None,
#         **dataset_configs.dataset_defaults[DATASET],
#     )
#     wilds_config.model_kwargs = {}
#     wilds_config.model = 'bert-base-uncased'
#     train_data, val_data, test_data, reweighting_data = get_data(args, wilds_config, DATASET)
#     train_loader = get_train_loader("standard", train_data, batch_size=args.batch_size, uniform_over_groups=False)
#     val_loader = get_eval_loader("standard", val_data, batch_size=args.batch_size)
#     test_loader = get_eval_loader("standard", test_data, batch_size=args.batch_size)
#     # reweighting_loader = get_eval_loader("standard", reweighting_data, batch_size=args.batch_size)
#
#     if reweight_flg == True:
#         with torch.no_grad():
#             all_train_logits, all_train_y_true = [], []
#             model.eval()
#             for batch in train_loader:
#                 input_ids = batch[0][:, :, 0].to(device)
#                 attention_mask = batch[0][:, :, 1].to(device)
#                 labels = batch[1].to(device)
#
#                 logits, _ = model(input_ids, attention_mask, labels)
#
#                 all_train_logits.append(logits)
#                 all_train_y_true.append(labels)
#
#             all_train_logits = torch.cat(all_train_logits, axis=0)
#             all_train_y_true = torch.cat(all_train_y_true, axis=0)
#
#             total_weights = compute_weights(all_train_logits, all_train_y_true, gamma, True)
#
#
#
#
#     for epoch in range(n_epochs):
#         model.train()
#         total_train_loss = 0
#         total_train_accuracy = 0
#         total_batches = 0
#         batch_start_idx = 0
#         batch_end_idx = 0
#         for batch in train_loader:
#             # inputs = tokenizer(batch['text'], return_tensors="pt", padding=True, truncation=True, max_length=512)
#             input_ids = batch[0][:, :, 0].to(device)
#             attention_mask = batch[0][:, :, 1].to(device)
#             labels = batch[1].to(device)
#             # print(input_ids.device)
#
#             batch_end_idx = batch_start_idx + len(labels)
#             weights = total_weights[batch_start_idx:batch_end_idx]  if total_weights is not None else None
#             batch_start_idx = batch_end_idx
#
#             logits, loss = model(input_ids, attention_mask, labels, weights)
#             loss.backward()
#             optimizer.step()
#             accuracy = compute_accuracy(logits, labels)
#             total_train_accuracy += accuracy.item()
#             total_train_loss += loss.item()
#             # print(f"Epoch {epoch}, Loss: {loss.item()}")
#             optimizer.zero_grad()
#
#             total_batches += 1
#             if total_batches % 50 == 0:
#                 print(f"Epoch {epoch}, batches {total_batches} , loss = {loss.item()}")
#
#         avg_train_loss = total_train_loss / len(train_loader)
#         avg_train_accuracy = total_train_accuracy / len(train_loader)
#         print(f"Epoch {epoch}, Train Loss: {avg_train_loss}, Train Accuracy: {avg_train_accuracy}")
#         wandb.log({"Train Loss": avg_train_loss, "Train Accuracy": avg_train_accuracy, "epoch": epoch})
#
#         model.eval()
#         val_results, total_val_loss, total_val_accuracy = evaluation(model, val_data, val_loader, device)
#         avg_val_loss = total_val_loss / len(val_loader)
#         avg_val_accuracy = total_val_accuracy / len(val_loader)
#         print(f"Epoch {epoch}, Validation Loss: {avg_val_loss}, Validation Accuracy: {avg_val_accuracy}")
#         print(val_results[1])
#         # wandb.log({"epoch": epoch, "Validation Loss": avg_val_loss, "Validation Accuracy": avg_val_accuracy})
#         wandb.log({"epoch": epoch, "Validation Loss": avg_val_loss, 'Validation Accuracy': val_results[0]['acc_avg'], 'Validation Worst Group Accuracy': val_results[0]['acc_wg']})
#
#         if val_results[0]['acc_wg'] > best_acc_wg:
#             best_accuracy = val_results[0]['acc_avg']
#             best_acc_wg = val_results[0]['acc_wg']
#             best_loss = avg_val_loss
#             torch.save(model.state_dict(), best_model_path)
#             print(f"Epoch {epoch}, Best Validation Loss: {best_loss}, Best Validation Accuracy: {best_accuracy}, Best Validation Worst Group Accuracy: {best_acc_wg}")
#             print("Saved best model")
#             wandb.log({"epoch": epoch, "Best Validation Loss": best_loss, "Best Validation Accuracy": best_accuracy, "Best Validation Worst Group Accuracy": best_acc_wg})
#
#     model.load_state_dict(torch.load(best_model_path, map_location=device))
#     model.eval()
#     test_results, total_test_loss, total_test_accuracy = evaluation(model, test_data, test_loader, device)
#
#     avg_test_loss = total_test_loss / len(test_loader)
#     avg_test_accuracy = total_test_accuracy / len(test_loader)
#     print(f"Test Loss: {avg_test_loss}, Test Accuracy: {avg_test_accuracy}")
#     print(test_results[1])
#     wandb.log({"Test Loss": avg_test_loss, "Test Accuracy": avg_test_accuracy})
#     wandb.log({'Test Mean Accuracy': test_results[0]['acc_avg'], 'Test Worst Group Accuracy': test_results[0]['acc_wg']})
#     wandb.log(test_results[0])
#     wandb.finish()
#
#
# if __name__ == '__main__':
# main()
# train_nli()
