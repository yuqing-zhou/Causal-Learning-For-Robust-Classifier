import random
import numpy as np
import torch

seeds = [42]

def set_seed(seed_value=42):
    random.seed(seed_value)  # Python random module
    np.random.seed(seed_value)  # Numpy module
    torch.manual_seed(seed_value)  # PyTorch
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def wandb_project_name(dataset, disentangle_version, pns_version, reweight_version):
    project_name = f"robust-learning-{dataset}"
    return project_name


def wandb_exp_name(DATASET_NAME, disentangle_version, pns_version, reweight_version, n_exp, lr, feature_size, reg_disentangle, reg_causal, gamma, weight_decay=0):
    name = f"{DATASET_NAME}_{n_exp}_disentangle_cov_reg_{disentangle_version}_{pns_version}_{reweight_version}_{feature_size}_{lr}_{reg_disentangle}_{reg_causal}_{gamma}_{weight_decay}"
    # name = f"{DATASET_NAME}_BERT_ERM_{disentangle_version}_{pns_version}_{reweight_version}_{feature_size}_{lr}_{reg_disentangle}_{reg_causal}_{gamma}_{weight_decay}"
    # name = f"{DATASET_NAME}_{n_exp}_no_disentangle_{disentangle_version}_{pns_version}_{reweight_version}_{feature_size}_{lr}_{reg_disentangle}_{reg_causal}_{gamma}_{weight_decay}" _stepLR_30_0.1
    # name = f"{DATASET_NAME}_{n_exp}_AFR_{disentangle_version}_{pns_version}_{reweight_version}_{feature_size}_{lr}_{reg_disentangle}_{reg_causal}_{gamma}_{weight_decay}"
    # name = f"{DATASET_NAME}_Bert_PNS_{disentangle_version}_{pns_version}_{reweight_version}_{feature_size}_{lr}_{reg_disentangle}_{reg_causal}_{gamma}_{weight_decay}"
    return name

def lr_lambda(epoch):
    return 0.8 ** (epoch)