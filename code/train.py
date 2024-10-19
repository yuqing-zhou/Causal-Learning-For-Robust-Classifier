import torch
import wandb
import os
from model import *
from utils import *
from settings import *
import models
import optimizers
import tqdm
from transformers import BertConfig, BertForSequenceClassification
from transforms_ import initialize_transform

_bar_format = '{l_bar}{bar:50}{r_bar}{bar:-10b}'

A = 1
B = 20  # 50 # 175 #50 #273 #205 # 25 #  125 # 225

C = 1
D = 50  # 70
E = 0  # 1

F = 22  # 19 #250 #200

DATASET_NAME = 'yelp-author-style'  # 'beer-concept-occurrence' #  # #'civilcomments'
JTT = True


def train_yelp(args):
    disentangle_version = 25  # args.disentangle_version
    pns_version = 1  # args.pns_version
    reweight_version = args.reweight_version
    n_exp = args.n_exp

    reg_disentangle = args.reg_disentangle
    lr = args.lr
    n_epochs = 15  # args.n_epochs
    batch_size = args.batch_size
    model_name = args.model_name
    dfr_reweighting_frac = args.dfr_reweighting_frac  # 0.2
    DATASET = args.dataset_name  # 'beer-corr' # 'beer-concept-occurrence'

    num_classes = DATASET_INFO[DATASET]['num_classes']
    feature_size = 512  # args.feature_size
    total_weights = None
    finetune_flg = args.finetune_flg  #
    reweight_flg = args.reweight_flg  #
    weight_decay = args.weight_decay

    seed = seeds[n_exp]
    root_dir = '../../data/'
    data_dir = root_dir + 'datasets/'
    model_save_path = root_dir + f'models/{DATASET}/'
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    best_model = None
    best_loss = float('inf')
    best_acc_wg = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    if finetune_flg == True:
        load_model_path = model_save_path + f'best_model_{disentangle_version}.pth'
        best_model_path = model_save_path + f'best_model_{disentangle_version}_{pns_version}_{reweight_version}.pth'
        # JTT
        # load_model_path = model_save_path + f'best_model_bert_ERM_{disentangle_version}.pth' # For JTT
        # best_model_path = model_save_path + f'best_model_JTT_{disentangle_version}_{pns_version}_{reweight_version}.pth'
        # AFR
        # load_model_path = model_save_path + f'model_bert_ERM_{disentangle_version}_epoch29.pth'
        # best_model_path = model_save_path + f'best_model_AFR_{disentangle_version}_{pns_version}_{reweight_version}.pth'

        # best_model_path = model_save_path + f'best_model_BERT_pns_{disentangle_version}_{pns_version}_{reweight_version}.pth'
        load_local_model = True
        reg_causal = args.reg_causal
        if args.reg_causal == 2:
            reg_causal = 0.001  # 12
        elif args.reg_causal == 3:
            reg_causal = 0.01  # 14
        elif args.reg_causal == 4:
            reg_causal = 0.1  # 16
        elif args.reg_causal == 5:
            reg_causal = 0.5  # 18S
        elif args.reg_causal == 6:
            reg_causal = 1  # 8

        disentangle_en = False
        counterfactual_en = True
    else:
        # best_model_path = model_save_path + f'best_model_{disentangle_version}.pth'
        # best_model_path = model_save_path + f'best_model_bert_ERM_{disentangle_version}.pth'
        # best_model_path = model_save_path + f'best_model_AFR_bert_ERM_{disentangle_version}.pth'
        load_local_model = False
        reg_causal = 0
        disentangle_en = True
        counterfactual_en = False

    if reweight_flg == True:
        gamma = args.gamma_reweight  #
        # if args.gamma_reweight == 0.001:
        #     gamma = 12 #12
        # elif args.gamma_reweight == 0.01:
        #     gamma = 14 #14
        # elif args.gamma_reweight == 0.1:
        #     gamma = 16 #16
        # elif args.gamma_reweight == 1:
        #     gamma = 18 #18S
        # elif args.gamma_reweight == 10:
        #     gamma = 20 #8
    else:
        gamma = 0

    project_name = wandb_project_name(DATASET, disentangle_version, pns_version, reweight_version)
    exp_name = wandb_exp_name(DATASET, disentangle_version, pns_version, reweight_version, n_exp, lr, feature_size,
                              reg_disentangle, reg_causal, gamma)
    wandb.init(
        # set the wandb project where this run will be logged
        project=project_name,  # "robust-learning",
        name=exp_name,  # f"disentangle_cov_reg_{disentangle_version}_{pns_version}_{reweight_version}",
        # notes="Bert with the regularization of the covariance matrix of the input of the last layer",
        # notes="Finetune the disentangled Bert, with the causality constraints, initialize the last later",
        notes="Finetune the disentangled Bert, with the causality constraints, initialize the last later, reweights the CE loss",
        # track hyperparameters and run metadata
        config={
            "learning_rate": lr,
            "architecture": model_name,
            "dataset": DATASET,
            "epochs": n_epochs,
            "batch_size": batch_size,
            "dfr_reweighting_frac": dfr_reweighting_frac,
            "Regularization coefficient": reg_disentangle,
            "Bert feature size": feature_size,
            "Causal Regularization coefficient": reg_causal,
            "gamma": gamma,
            "seed": seed,
            "weight_decay": weight_decay,
        }
    )
    wandb.define_metric("epoch")
    wandb.define_metric("Train Loss", step_metric='epoch')
    wandb.define_metric("Train Accuracy", step_metric='epoch')
    wandb.define_metric("Validation Loss", step_metric='epoch')
    wandb.define_metric("Validation Accuracy", step_metric='epoch')
    wandb.define_metric("Best Validation Loss", step_metric='epoch')
    wandb.define_metric("Best Validation Accuracy", step_metric='epoch')
    wandb.define_metric("Best Validation Worst Group Accuracy", step_metric='epoch')

    set_seed(seed)
    model = BertClassifierWithCovReg(model_name, num_labels=num_classes, feature_size=feature_size, device=device,
                                     reg=reg_disentangle, reg_causal=reg_causal, disentangle_en=disentangle_en,
                                     counterfactual_en=counterfactual_en).to(device)
    if load_local_model:
        model.load_state_dict(torch.load(load_model_path, map_location=device))
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    dataset_configs.dataset_defaults[DATASET]['batch_size'] = batch_size
    task_config = SimpleNamespace(
        root_dir=data_dir,
        # batch_size=batch_size,
        dfr_reweighting_drop=True,
        dfr_reweighting_seed=seed,
        dfr_reweighting_frac=dfr_reweighting_frac,
        algorithm='ERM',
        load_featurizer_only=False,
        pretrained_model_path=None,
        **dataset_configs.dataset_defaults[DATASET],
    )

    task_config.model_kwargs = {}
    task_config.model = 'bert-base-uncased'

    transform = initialize_transform(
        transform_name=task_config.transform,
        config=task_config,
        dataset=None,
        additional_transform_name=None,
        is_training=False)
    train_loader = load_data(os.path.join(data_dir, DATASET_INFO[DATASET]['dataset_path'], 'train.csv'), batch_size,
                             transform)
    val_loader = load_data(os.path.join(data_dir, DATASET_INFO[DATASET]['dataset_path'], 'val.csv'), batch_size,
                           transform)
    test_loader = load_data(os.path.join(data_dir, DATASET_INFO[DATASET]['dataset_path'], 'test.csv'), batch_size,
                            transform)
    anti_test_loader = load_data(
        os.path.join(data_dir, DATASET_INFO[DATASET]['dataset_path'], 'test_anti-shortcut.csv'), batch_size, transform)

    if reweight_flg == True:
        with torch.no_grad():
            all_train_logits, all_train_y_true = [], []
            model.eval()
            for batch in train_loader:
                input_ids = batch[0][:, :, 0].to(device)
                attention_mask = batch[0][:, :, 1].to(device)
                labels = batch[1].to(device)

                logits, _ = model(input_ids, attention_mask, labels)

                all_train_logits.append(logits)
                all_train_y_true.append(labels)

            all_train_logits = torch.cat(all_train_logits, axis=0)
            all_train_y_true = torch.cat(all_train_y_true, axis=0)

            total_weights = compute_weights1(all_train_logits, all_train_y_true, gamma, True)

    # if JTT:
    #     del model
    #     model = BertClassifierWithCovReg(model_name, num_labels=num_classes, feature_size=feature_size, device=device,
    #                                  reg=reg_disentangle, reg_causal=reg_causal, disentangle_en=disentangle_en,
    #                                  counterfactual_en=counterfactual_en).to(device)
    # else:
    if load_local_model:
        model = model_parameters_freeze(model)

    for epoch in range(n_epochs):
        model.train()
        total_train_loss = 0
        total_train_accuracy = 0
        total_batches = 0
        batch_start_idx = 0
        batch_end_idx = 0
        for batch in train_loader:
            # inputs = tokenizer(batch['text'], return_tensors="pt", padding=True, truncation=True, max_length=512)
            input_ids = batch[0][:, :, 0].to(device)
            attention_mask = batch[0][:, :, 1].to(device)
            labels = batch[1].to(device)
            # print(input_ids.device)

            batch_end_idx = batch_start_idx + len(labels)
            weights = total_weights[batch_start_idx:batch_end_idx] if total_weights is not None else None
            batch_start_idx = batch_end_idx

            logits, loss = model(input_ids, attention_mask, labels, weights)
            loss.backward()
            optimizer.step()
            accuracy = compute_accuracy(logits, labels)
            total_train_accuracy += accuracy.item()
            total_train_loss += loss.item()
            # print(f"Epoch {epoch}, Loss: {loss.item()}")
            optimizer.zero_grad()

            total_batches += 1
            if total_batches % 50 == 0:
                print(f"Epoch {epoch}, batches {total_batches} , loss = {loss.item()}")

        avg_train_loss = total_train_loss / total_batches
        avg_train_accuracy = total_train_accuracy / total_batches
        print(f"Epoch {epoch}, Train Loss: {avg_train_loss}, Train Accuracy: {avg_train_accuracy}")
        wandb.log({"Train Loss": avg_train_loss, "Train Accuracy": avg_train_accuracy, "epoch": epoch})

        model.eval()
        val_group_acc, avg_val_loss, avg_val_accuracy, _ = evaluation_nli(model, 4, val_loader, device)
        acc_wg_val, _ = torch.min(val_group_acc, dim=0)
        print(
            f"Epoch {epoch}, Validation Loss: {avg_val_loss}, Validation Accuracy: {avg_val_accuracy}, Validation Worst Group Accuracy: {acc_wg_val}")
        # wandb.log({"epoch": epoch, "Validation Loss": avg_val_loss, "Validation Accuracy": avg_val_accuracy})
        wandb.log({"epoch": epoch, "Validation Loss": avg_val_loss, 'Validation Accuracy': avg_val_accuracy,
                   'Validation Worst Group Accuracy': acc_wg_val})

        if acc_wg_val.item() >= best_acc_wg:
            best_accuracy = avg_val_accuracy
            best_acc_wg = acc_wg_val.item()
            best_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            print(
                f"Epoch {epoch}, Best Validation Loss: {best_loss}, Best Validation Accuracy: {best_accuracy}, Best Validation Worst Group Accuracy: {best_acc_wg}")
            print("Saved best model")
            wandb.log({"epoch": epoch, "Best Validation Loss": best_loss, "Best Validation Accuracy": best_accuracy,
                       "Best Validation Worst Group Accuracy": best_acc_wg})

    # final_model_path = model_save_path + f'model_bert_ERM_{disentangle_version}_epoch{epoch}.pth'
    # torch.save(model.state_dict(), final_model_path)
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()
    test_group_acc, avg_test_loss, avg_test_accuracy, _ = evaluation_nli(model, 4, test_loader, device)
    acc_wg_test, _ = torch.min(test_group_acc, dim=0)
    print(f"Test Loss: {avg_test_loss}, Test Accuracy: {avg_test_accuracy}, Test Worst Group Acc: {acc_wg_test.item()}")
    wandb.log({"Test Loss": avg_test_loss, "Test Accuracy": avg_test_accuracy})
    wandb.log({'Test Mean Accuracy': avg_test_accuracy, 'Test Worst Group Accuracy': acc_wg_test.item()})

    test_group_acc, avg_test_loss, avg_test_accuracy, _ = evaluation_nli(model, 4, anti_test_loader, device)
    acc_wg_test, _ = torch.min(test_group_acc, dim=0)
    print(
        f"Anti-test Loss: {avg_test_loss}, Anti-test Accuracy: {avg_test_accuracy}, Anti-test Worst Group Acc: {acc_wg_test.item()}")
    wandb.log({"Anti-test Loss": avg_test_loss, "Anti-test Accuracy": avg_test_accuracy})
    wandb.log({'Anti-test Mean Accuracy': avg_test_accuracy, 'Anti-test Worst Group Accuracy': acc_wg_test.item()})

    wandb.finish()


def train_civil(args):
    disentangle_version = 47  # args.disentangle_version
    pns_version = args.pns_version
    reweight_version = args.reweight_version
    n_exp = args.n_exp

    reg_disentangle = args.reg_disentangle
    lr = 2e-5  # args.lr / 3 * 2
    n_epochs = 10  # args.n_epochs
    batch_size = args.batch_size
    model_name = args.model_name
    dfr_reweighting_frac = args.dfr_reweighting_frac  # 0.2
    DATASET = args.dataset_name

    num_classes = DATASET_INFO[DATASET]['num_classes']
    feature_size = 128  # args.feature_size
    total_weights = None
    finetune_flg = args.finetune_flg  # True #
    reweight_flg = args.reweight_version  # True
    weight_decay = args.weight_decay

    seed = seeds[n_exp]
    root_dir = '../../data/'
    data_dir = root_dir + 'datasets/'
    model_save_path = root_dir + f'models/{DATASET}/'
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    best_model = None
    best_loss = float('inf')
    best_acc_wg = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    if finetune_flg == True:
        load_model_path = model_save_path + f'best_model_{disentangle_version}.pth'
        best_model_path = model_save_path + f'best_model_{disentangle_version}_{pns_version}_{reweight_version}.pth'
        load_local_model = True
        reg_causal = args.reg_causal
        disentangle_en = False
        counterfactual_en = True
    else:
        best_model_path = model_save_path + f'best_model_{disentangle_version}.pth'
        load_local_model = False
        reg_causal = 0
        disentangle_en = True
        counterfactual_en = False

    if reweight_flg == True:
        gamma = args.gamma_reweight
    else:
        gamma = 0

    project_name = wandb_project_name(DATASET, disentangle_version, pns_version, reweight_version)
    exp_name = wandb_exp_name(DATASET, disentangle_version, pns_version, reweight_version, n_exp, lr, feature_size,
                              reg_disentangle, reg_causal, gamma)
    wandb.init(
        # set the wandb project where this run will be logged
        project=project_name,  # "robust-learning",
        name=exp_name,  # f"disentangle_cov_reg_{disentangle_version}_{pns_version}_{reweight_version}",
        # notes="Bert with the regularization of the covariance matrix of the input of the last layer",
        # notes="Finetune the disentangled Bert, with the causality constraints, initialize the last later",
        notes="Finetune the disentangled Bert, with the causality constraints, initialize the last later, reweights the CE loss",
        # track hyperparameters and run metadata
        config={
            "learning_rate": lr,
            "architecture": model_name,
            "dataset": DATASET,
            "epochs": n_epochs,
            "batch_size": batch_size,
            "dfr_reweighting_frac": dfr_reweighting_frac,
            "Regularization coefficient": reg_disentangle,
            "Bert feature size": feature_size,
            "Causal Regularization coefficient": reg_causal,
            "gamma": gamma,
            "seed": seed,
            "weight_decay": weight_decay,
        }
    )
    wandb.define_metric("epoch")
    wandb.define_metric("Train Loss", step_metric='epoch')
    wandb.define_metric("Train Accuracy", step_metric='epoch')
    wandb.define_metric("Validation Loss", step_metric='epoch')
    wandb.define_metric("Validation Accuracy", step_metric='epoch')
    wandb.define_metric("Best Validation Loss", step_metric='epoch')
    wandb.define_metric("Best Validation Accuracy", step_metric='epoch')
    wandb.define_metric("Best Validation Worst Group Accuracy", step_metric='epoch')

    set_seed(seed)
    model = BertClassifierWithCovReg(model_name, num_labels=num_classes, feature_size=feature_size, device=device,
                                     reg=reg_disentangle, reg_causal=reg_causal, disentangle_en=disentangle_en,
                                     counterfactual_en=counterfactual_en).to(device)
    if load_local_model:
        model.load_state_dict(torch.load(load_model_path, map_location=device))
        model = model_parameters_freeze(model)
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # args = SimpleNamespace(
    #     root_dir=data_dir,
    #     batch_size=batch_size,
    #     dfr_reweighting_drop=True,
    #     dfr_reweighting_seed=seed,
    #     dfr_reweighting_frac=dfr_reweighting_frac,
    # )
    # wilds_config = SimpleNamespace(
    #     algorithm='ERM',
    #     load_featurizer_only=False,
    #     pretrained_model_path=None,
    #     **dataset_configs.dataset_defaults[DATASET],
    # )
    dataset_configs.dataset_defaults[DATASET]['batch_size'] = batch_size
    task_config = SimpleNamespace(
        root_dir=data_dir,
        # batch_size=batch_size,
        dfr_reweighting_drop=True,
        dfr_reweighting_seed=seed,
        dfr_reweighting_frac=dfr_reweighting_frac,
        algorithm='ERM',
        load_featurizer_only=False,
        pretrained_model_path=None,
        **dataset_configs.dataset_defaults[DATASET],
    )

    task_config.model_kwargs = {}
    task_config.model = 'bert-base-uncased'
    train_data, val_data, test_data, reweighting_data = get_data(task_config, DATASET)
    train_loader = get_train_loader("standard", train_data, batch_size=batch_size, uniform_over_groups=False)
    val_loader = get_eval_loader("standard", val_data, batch_size=batch_size)
    test_loader = get_eval_loader("standard", test_data, batch_size=batch_size)
    # reweighting_loader = get_eval_loader("standard", reweighting_data, batch_size=args.batch_size)

    if reweight_flg == True:
        with torch.no_grad():
            all_train_logits, all_train_y_true = [], []
            model.eval()
            for batch in train_loader:
                input_ids = batch[0][:, :, 0].to(device)
                attention_mask = batch[0][:, :, 1].to(device)
                labels = batch[1].to(device)

                logits, _ = model(input_ids, attention_mask, labels)

                all_train_logits.append(logits)
                all_train_y_true.append(labels)

            all_train_logits = torch.cat(all_train_logits, axis=0)
            all_train_y_true = torch.cat(all_train_y_true, axis=0)

            total_weights = compute_weights(all_train_logits, all_train_y_true, gamma, True)

    for epoch in range(n_epochs):
        model.train()
        total_train_loss = 0
        total_train_accuracy = 0
        total_batches = 0
        batch_start_idx = 0
        batch_end_idx = 0
        for batch in train_loader:
            # inputs = tokenizer(batch['text'], return_tensors="pt", padding=True, truncation=True, max_length=512)
            input_ids = batch[0][:, :, 0].to(device)
            attention_mask = batch[0][:, :, 1].to(device)
            labels = batch[1].to(device)
            # print(input_ids.device)

            batch_end_idx = batch_start_idx + len(labels)
            weights = total_weights[batch_start_idx:batch_end_idx] if total_weights is not None else None
            batch_start_idx = batch_end_idx

            logits, loss = model(input_ids, attention_mask, labels, weights)
            loss.backward()
            optimizer.step()
            accuracy = compute_accuracy(logits, labels)
            total_train_accuracy += accuracy.item()
            total_train_loss += loss.item()
            # print(f"Epoch {epoch}, Loss: {loss.item()}")
            optimizer.zero_grad()

            total_batches += 1
            if total_batches % 50 == 0:
                print(f"Epoch {epoch}, batches {total_batches} , loss = {loss.item()}")

        avg_train_loss = total_train_loss / total_batches
        avg_train_accuracy = total_train_accuracy / total_batches
        print(f"Epoch {epoch}, Train Loss: {avg_train_loss}, Train Accuracy: {avg_train_accuracy}")
        wandb.log({"Train Loss": avg_train_loss, "Train Accuracy": avg_train_accuracy, "epoch": epoch})

        model.eval()
        val_results, avg_val_loss, avg_val_accuracy = evaluation(model, val_data, val_loader, device)
        print(f"Epoch {epoch}, Validation Loss: {avg_val_loss}, Validation Accuracy: {avg_val_accuracy}")
        print(val_results[1])
        # wandb.log({"epoch": epoch, "Validation Loss": avg_val_loss, "Validation Accuracy": avg_val_accuracy})
        wandb.log({"epoch": epoch, "Validation Loss": avg_val_loss, 'Validation Accuracy': val_results[0]['acc_avg'],
                   'Validation Worst Group Accuracy': val_results[0]['acc_wg']})

        if val_results[0]['acc_wg'] > best_acc_wg:
            best_accuracy = val_results[0]['acc_avg']
            best_acc_wg = val_results[0]['acc_wg']
            best_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            print(
                f"Epoch {epoch}, Best Validation Loss: {best_loss}, Best Validation Accuracy: {best_accuracy}, Best Validation Worst Group Accuracy: {best_acc_wg}")
            print("Saved best model")
            wandb.log({"epoch": epoch, "Best Validation Loss": best_loss, "Best Validation Accuracy": best_accuracy,
                       "Best Validation Worst Group Accuracy": best_acc_wg})

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()
    test_results, avg_test_loss, avg_test_accuracy = evaluation(model, test_data, test_loader, device)
    print(f"Test Loss: {avg_test_loss}, Test Accuracy: {avg_test_accuracy}")
    print(test_results[1])
    wandb.log({"Test Loss": avg_test_loss, "Test Accuracy": avg_test_accuracy})
    wandb.log(
        {'Test Mean Accuracy': test_results[0]['acc_avg'], 'Test Worst Group Accuracy': test_results[0]['acc_wg']})
    wandb.log(test_results[0])
    wandb.finish()


def train_nli(args):
    disentangle_version = args.disentangle_version
    pns_version = args.pns_version
    reweight_version = args.reweight_version
    n_exp = args.n_exp
    seed = seeds[n_exp]
    set_seed(seed)

    reg_disentangle = args.reg_disentangle
    lr = args.lr
    n_epochs = args.n_epochs
    batch_size = args.batch_size
    model_name = args.model_name
    dfr_reweighting_frac = args.dfr_reweighting_frac  # 0.2

    DATASET = args.dataset_name

    num_classes = DATASET_INFO[DATASET]['num_classes']
    feature_size = args.feature_size
    finetune_flg = args.finetune_flg  # True #
    reweight_flg = args.reweight_version  # True
    weight_decay = args.weight_decay

    root_dir = '../../data/'
    if DATASET == 'civilcomments':
        data_dir = root_dir + 'datasets/'
    elif DATASET == 'MultiNLI':
        data_dir = root_dir + 'datasets/multinli_bert_features/'
    model_save_path = root_dir + f'models/{DATASET}/'
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    total_weights = None
    best_model = None
    best_loss = float('inf')
    best_acc_wg = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    if finetune_flg == True:
        load_model_path = model_save_path + f'best_model_{disentangle_version}.pth'
        best_model_path = model_save_path + f'best_model_{disentangle_version}_{pns_version}_{reweight_version}.pth'
        load_local_model = True
        reg_causal = args.reg_causal
        disentangle_en = False
        counterfactual_en = True
    else:
        best_model_path = model_save_path + f'best_model_{disentangle_version}.pth'
        load_local_model = False
        reg_causal = 0
        disentangle_en = True
        counterfactual_en = False

    if reweight_flg == True:
        gamma = args.gamma_reweight
    else:
        gamma = 0

    project_name = wandb_project_name(DATASET, disentangle_version, pns_version, reweight_version)
    exp_name = wandb_exp_name(DATASET, disentangle_version, pns_version, reweight_version, n_exp, lr, feature_size,
                              reg_disentangle, reg_causal, gamma)
    wandb.init(
        # set the wandb project where this run will be logged
        project=project_name,  # "robust-learning",
        name=exp_name,  # f"disentangle_cov_reg_{disentangle_version}_{pns_version}_{reweight_version}",
        # notes="Bert with the regularization of the covariance matrix of the input of the last layer",
        # notes="Finetune the disentangled Bert, with the causality constraints, initialize the last later",
        notes="Finetune the disentangled Bert, with the causality constraints, initialize the last later, reweights the CE loss",
        # track hyperparameters and run metadata
        config={
            "learning_rate": lr,
            "architecture": model_name,
            "dataset": DATASET,
            "epochs": n_epochs,
            "batch_size": batch_size,
            "dfr_reweighting_frac": dfr_reweighting_frac,
            "Regularization coefficient": reg_disentangle,
            "Bert feature size": feature_size,
            "Causal Regularization coefficient": reg_causal,
            "gamma": gamma,
            "seed": seed,
            "weight_decay": weight_decay,
        }
    )
    wandb.define_metric("epoch")
    wandb.define_metric("Train Loss", step_metric='epoch')
    wandb.define_metric("Train Accuracy", step_metric='epoch')
    wandb.define_metric("Validation Loss", step_metric='epoch')
    wandb.define_metric("Validation Accuracy", step_metric='epoch')
    wandb.define_metric("Best Validation Loss", step_metric='epoch')
    wandb.define_metric("Best Validation Accuracy", step_metric='epoch')
    wandb.define_metric("Best Validation Worst Group Accuracy", step_metric='epoch')

    model = BertClassifierWithCovReg(model_name, num_labels=num_classes, feature_size=feature_size, device=device,
                                     reg=reg_disentangle, reg_causal=reg_causal, disentangle_en=disentangle_en,
                                     counterfactual_en=counterfactual_en).to(device)
    # # model = Bert(model_name, num_labels=num_classes, feature_size=feature_size, device=device, reg=reg_disentangle,
    # #              reg_causal=reg_causal, disentangle_en=disentangle_en, counterfactual_en=counterfactual_en).to(device)
    # config_class = BertConfig
    # model_class = BertForSequenceClassification
    # config = config_class.from_pretrained(
    #             'bert-base-uncased',
    #             num_labels=num_classes,
    #             finetuning_task='mnli')
    # model = BertForSequenceClassificationWithCovReg.from_pretrained(
    #             'bert-base-uncased',
    #             config=config,
    #             feature_size=feature_size,
    #             device=device,
    #             reg_disentangle=reg_disentangle,
    #             reg_causal=reg_causal,
    #             disentangle_en=disentangle_en,
    #             counterfactual_en=counterfactual_en).to(device)
    if load_local_model:
        model.load_state_dict(torch.load(load_model_path, map_location=device))
        model = model_parameters_freeze(model)
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    if DATASET == 'civilcomments':
        dataset_configs.dataset_defaults[DATASET]['batch_size'] = batch_size
        task_config = SimpleNamespace(
            root_dir=data_dir,
            # batch_size=batch_size,
            dfr_reweighting_drop=True,
            dfr_reweighting_seed=seed,
            dfr_reweighting_frac=dfr_reweighting_frac,
            algorithm='ERM',
            load_featurizer_only=False,
            pretrained_model_path=None,
            **dataset_configs.dataset_defaults[DATASET],
        )
        task_config.model_kwargs = {}
        task_config.model = model_name
    else:
        task_config = SimpleNamespace(
            root_dir=data_dir,
            batch_size=batch_size,
            dataset=DATASET,
            reweight_groups=False,
            target_name='gold_label_random',
            confounder_names=['sentence2_has_negation'],
            model='bert',
            augment_data=False,
            fraction=1.0,
        )
    train_data, val_data, test_data = get_data(task_config, DATASET, train=True)

    loader_kwargs = {'batch_size': batch_size, 'num_workers': 4, 'pin_memory': True}
    train_loader = get_data_loader(DATASET, train_data, task_config, train=True, **loader_kwargs)
    val_loader = get_data_loader(DATASET, val_data, task_config, train=False, **loader_kwargs)
    test_loader = get_data_loader(DATASET, test_data, task_config, train=False, **loader_kwargs)

    data = {}
    data['train_loader'] = train_loader
    data['val_loader'] = val_loader
    data['test_loader'] = test_loader
    data['train_data'] = train_data
    data['val_data'] = val_data
    data['test_data'] = test_data

    if reweight_flg == True:
        with torch.no_grad():
            all_train_logits, all_train_y_true = [], []
            model.eval()
            for batch in train_loader:
                input_ids = batch[0][:, :, 0].to(device)
                attention_mask = batch[0][:, :, 1].to(device)
                segment_ids = batch[0][:, :, 2].to(device)
                labels = batch[1].to(device)

                logits, _ = model(input_ids, attention_mask, labels, token_type_ids=segment_ids)

                all_train_logits.append(logits)
                all_train_y_true.append(labels)

            all_train_logits = torch.cat(all_train_logits, axis=0)
            all_train_y_true = torch.cat(all_train_y_true, axis=0)

            total_weights = compute_weights(all_train_logits, all_train_y_true, gamma, True)

    for epoch in range(n_epochs):
        model.train()
        total_train_loss = 0
        total_train_accuracy = 0
        total_batches = 0
        batch_start_idx = 0
        batch_end_idx = 0
        for batch in train_loader:
            # inputs = tokenizer(batch['text'], return_tensors="pt", padding=True, truncation=True, max_length=512)
            input_ids = batch[0][:, :, 0].to(device)
            attention_mask = batch[0][:, :, 1].to(device)
            segment_ids = batch[0][:, :, 2].to(device)
            labels = batch[1].to(device)
            # print(input_ids.device)

            batch_end_idx = batch_start_idx + len(labels)
            weights = total_weights[batch_start_idx:batch_end_idx] if total_weights is not None else None
            batch_start_idx = batch_end_idx

            logits, loss = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=segment_ids,
                                 labels=labels, weights=weights)
            loss.backward()
            optimizer.step()
            accuracy = compute_accuracy(logits, labels)
            total_train_accuracy += accuracy.item()
            total_train_loss += loss.item()
            # print(f"Epoch {epoch}, Loss: {loss.item()}")
            optimizer.zero_grad()

            total_batches += 1
            if total_batches % 50 == 0:
                print(f"Epoch {epoch}, batches {total_batches} , loss = {loss.item()}")

        avg_train_loss = total_train_loss / total_batches
        avg_train_accuracy = total_train_accuracy / total_batches
        print(f"Epoch {epoch}, Train Loss: {avg_train_loss}, Train Accuracy: {avg_train_accuracy}")
        wandb.log({"Train Loss": avg_train_loss, "Train Accuracy": avg_train_accuracy, "epoch": epoch})

        model.eval()
        val_group_acc, avg_val_loss, avg_val_accuracy, _ = evaluation_nli(model, val_data.n_groups, val_loader, device)
        acc_wg_val, _ = torch.min(val_group_acc, dim=0)
        print(
            f"Epoch {epoch}, Validation Loss: {avg_val_loss}, Validation Accuracy: {avg_val_accuracy}, Validation Worst Group Accuracy: {acc_wg_val}")
        # wandb.log({"epoch": epoch, "Validation Loss": avg_val_loss, "Validation Accuracy": avg_val_accuracy})
        wandb.log({"epoch": epoch, "Validation Loss": avg_val_loss, 'Validation Accuracy': avg_val_accuracy,
                   'Validation Worst Group Accuracy': acc_wg_val})

        if acc_wg_val.item() >= best_acc_wg:
            best_accuracy = avg_val_accuracy
            best_acc_wg = acc_wg_val.item()
            best_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            print(
                f"Epoch {epoch}, Best Validation Loss: {best_loss}, Best Validation Accuracy: {best_accuracy}, Best Validation Worst Group Accuracy: {best_acc_wg}")
            print("Saved best model")
            wandb.log({"epoch": epoch, "Best Validation Loss": best_loss, "Best Validation Accuracy": best_accuracy,
                       "Best Validation Worst Group Accuracy": best_acc_wg})

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()
    test_group_acc, avg_test_loss, avg_test_accuracy, _ = evaluation_nli(model, test_data.n_groups, test_loader, device)
    acc_wg_test, _ = torch.min(test_group_acc, dim=0)
    print(f"Test Loss: {avg_test_loss}, Test Accuracy: {avg_test_accuracy}, Test Worst Group Acc: {acc_wg_test.item()}")
    wandb.log({"Test Loss": avg_test_loss, "Test Accuracy": avg_test_accuracy})
    wandb.log({'Test Mean Accuracy': avg_test_accuracy, 'Test Worst Group Accuracy': acc_wg_test.item()})
    wandb.finish()

