import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertForSequenceClassification

from typing import List, Optional, Tuple, Union
from transformers.modeling_outputs import  SequenceClassifierOutput


model_attributes = {
    'bert': {
        'feature_type': 'text'
    },
    'inception_v3': {
        'feature_type': 'image',
        'target_resolution': (299, 299),
        'flatten': False
    },
    'wideresnet50': {
        'feature_type': 'image',
        'target_resolution': (224, 224),
        'flatten': False
    },
    'resnet50': {
        'feature_type': 'image',
        'target_resolution': (224, 224),
        'flatten': False
    },
    'resnet34': {
        'feature_type': 'image',
        'target_resolution': None,
        'flatten': False
    },
    'raw_logistic_regression': {
        'feature_type': 'image',
        'target_resolution': None,
        'flatten': True,
    }
}


class BertForSequenceClassificationWithCovReg(BertForSequenceClassification):
    def __init__(self, config, feature_size, device, reg_disentangle=0, reg_causal=0, disentangle_en=False,
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
        # logits = self.classifier(pooled_output)

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
                print('causal_reg=', causal_regularization, '\n')
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
                ##################################################################
                # Modified
                loss_fct = nn.CrossEntropyLoss(reduction='none')
                print("Problem type is single_label_classification.")
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                if causal_regularization is not None:
                    print('Add causal constraints ', self.reg_causal, '\n')
                    loss += self.reg_causal * causal_regularization
                if weights is not None:
                    print('Reweight\n')
                    loss *= weights
                loss = loss.mean() + self.reg_disentangle * regularization
                ###################################################################
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


# class BertForSequenceClassificationWithCovReg(BertForSequenceClassification):
#     def __init__(self, config, feature_size, device, reg_disentangle=0, reg_causal=0, disentangle_en=False,
#                  counterfactual_en=False):
#         super().__init__(config)
    # def __init__(self, config):
    #     super().__init__(config)

    #     self.bert = BertModel(config)
    #     classifier_dropout = (
    #         config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
    #     )
    #     self.dropout = nn.Dropout(classifier_dropout)
    #     self.classifier = nn.Linear(config.hidden_size, 1)

    #     # Initialize weights and apply final processing
    #     self.post_init()

    # @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length"))
    # @add_code_sample_docstrings(
    #     checkpoint=_CHECKPOINT_FOR_DOC,
    #     output_type=MultipleChoiceModelOutput,
    #     config_class=_CONFIG_FOR_DOC,
    # )
    # def forward(
    #     self,
    #     input_ids: Optional[torch.Tensor] = None,
    #     attention_mask: Optional[torch.Tensor] = None,
    #     token_type_ids: Optional[torch.Tensor] = None,
    #     position_ids: Optional[torch.Tensor] = None,
    #     head_mask: Optional[torch.Tensor] = None,
    #     inputs_embeds: Optional[torch.Tensor] = None,
    #     labels: Optional[torch.Tensor] = None,
    #     output_attentions: Optional[bool] = None,
    #     output_hidden_states: Optional[bool] = None,
    #     return_dict: Optional[bool] = None,
    # ) -> Union[Tuple[torch.Tensor], MultipleChoiceModelOutput]:
    #     r"""
    #     labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
    #         Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
    #         num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
    #         `input_ids` above)
    #     """
    #     return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    #     num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

    #     input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
    #     attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
    #     token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
    #     position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
    #     inputs_embeds = (
    #         inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
    #         if inputs_embeds is not None
    #         else None
    #     )

    #     outputs = self.bert(
    #         input_ids,
    #         attention_mask=attention_mask,
    #         token_type_ids=token_type_ids,
    #         position_ids=position_ids,
    #         head_mask=head_mask,
    #         inputs_embeds=inputs_embeds,
    #         output_attentions=output_attentions,
    #         output_hidden_states=output_hidden_states,
    #         return_dict=return_dict,
    #     )

    #     pooled_output = outputs[1]

    #     pooled_output = self.dropout(pooled_output)
    #     logits = self.classifier(pooled_output)
    #     reshaped_logits = logits.view(-1, num_choices)

    #     loss = None
    #     if labels is not None:
    #         loss_fct = CrossEntropyLoss()
    #         loss = loss_fct(reshaped_logits, labels)

    #     if not return_dict:
    #         output = (reshaped_logits,) + outputs[2:]
    #         return ((loss,) + output) if loss is not None else output

    #     return MultipleChoiceModelOutput(
    #         loss=loss,
    #         logits=reshaped_logits,
    #         hidden_states=outputs.hidden_states,
    #         attentions=outputs.attentions,
    #     )

def model_parameters_freeze(model):
    # Freeze layers of the pretrained model except the last linear layer
    for name, param in model.named_parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        nn.init.normal_(param, mean=0, std=1)
        param.requires_grad = True

    print("\nAfter fixing the layers before the last linear layer:")
    for name, param in model.named_parameters():
        print(name, param.requires_grad)

    return model

def compute_weights(logits, labels, gamma, balance_classes=True, all_group_ids=None):
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

def compute_weights1(logits, labels, gamma, balance_classes=True, all_group_ids=None):
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


def compute_weights2(logits, labels, gamma, balance_classes=True, all_group_ids=None):
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