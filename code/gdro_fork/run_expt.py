import os, csv
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torchvision

from models import model_attributes, BertForSequenceClassificationWithCovReg, model_parameters_freeze
from data.data import dataset_attributes, shift_types, prepare_data, log_data
from utils import set_seed, Logger, CSVBatchLogger, log_args
from train import train

import numpy as np


def main():
    parser = argparse.ArgumentParser()

    # Settings
    parser.add_argument('-d', '--dataset', choices=dataset_attributes.keys(), required=True)
    parser.add_argument('-s', '--shift_type', choices=shift_types, required=True)
    # Confounders
    parser.add_argument('-t', '--target_name')
    parser.add_argument('-c', '--confounder_names', nargs='+')
    # Resume?
    parser.add_argument('--resume', default=False, action='store_true')
    # Label shifts
    parser.add_argument('--minority_fraction', type=float)
    parser.add_argument('--imbalance_ratio', type=float)
    # Data
    parser.add_argument('--fraction', type=float, default=1.0)
    parser.add_argument('--root_dir', default=None)
    parser.add_argument('--reweight_groups', action='store_true', default=False)
    parser.add_argument('--augment_data', action='store_true', default=False)
    parser.add_argument('--val_fraction', type=float, default=0.1)
    # Objective
    parser.add_argument('--robust', default=False, action='store_true')
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--generalization_adjustment', default="0.0")
    parser.add_argument('--automatic_adjustment', default=False, action='store_true')
    parser.add_argument('--robust_step_size', default=0.01, type=float)
    parser.add_argument('--use_normalized_loss', default=False, action='store_true')
    parser.add_argument('--btl', default=False, action='store_true')
    parser.add_argument('--hinge', default=False, action='store_true')

    # Model
    parser.add_argument(
        '--model',
        choices=model_attributes.keys(),
        default='resnet50')
    parser.add_argument('--train_from_scratch', action='store_true', default=False)

    # Optimization
    parser.add_argument('--n_epochs', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--scheduler', action='store_true', default=False)
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--minimum_variational_weight', type=float, default=0)
    # Misc
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--show_progress', default=False, action='store_true')
    parser.add_argument('--log_dir', default='./logs')
    parser.add_argument('--log_every', default=50, type=int)
    parser.add_argument('--save_step', type=int, default=10)
    parser.add_argument('--save_best', action='store_true', default=False)
    parser.add_argument('--save_last', action='store_true', default=False)

    ####################################################################################################################
    # DFR custom args
    parser.add_argument('--dfr_reweighting_drop', action='store_true', default=False,
                        help='Remove a dedicated reweighting data split from train data.')
    parser.add_argument('--dfr_reweighting_seed', default=0, type=int,
                        help='Random seed used for the reweighting data split.')
    parser.add_argument('--dfr_reweighting_frac', type=float, default=0.2,
                        help='Fraction of data to remove as a dedicated DFR reweighting data split.')
    ####################################################################################################################

    parser.add_argument('--features_size', type=int, default=128)
    parser.add_argument('--reg_disentangle', type=float, default=0,
                        help='Coefficient of the regularization term for disentanglement.')
    parser.add_argument('--reg_causal', type=float, default=0,
                        help='Coefficient of the causal regularization term')
    parser.add_argument('--disentangle_en', default=False, action='store_true')
    parser.add_argument('--counterfactual_en', default=False, action='store_true')
    parser.add_argument('--finetune_en', default=False, action='store_true')
    parser.add_argument('--reweight_en', default=False, action='store_true')
    parser.add_argument('--load_model_path', default='./logs')

    args = parser.parse_args()
    check_args(args)

    # BERT-specific configs copied over from run_glue.py
    if args.model == 'bert':
        args.max_grad_norm = 1.0
        args.adam_epsilon = 1e-8
        args.warmup_steps = 0

    if os.path.exists(args.log_dir) and args.resume:
        resume = True
        mode = 'a'
    else:
        resume = False
        mode = 'w'

    ## Initialize logs
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    logger = Logger(os.path.join(args.log_dir, 'log.txt'), mode)
    # Record args
    log_args(args, logger)

    set_seed(args.seed)

    # Data
    # Test data for label_shift_step is not implemented yet
    test_data = None
    test_loader = None
    if args.shift_type == 'confounder':
        train_data, val_data, test_data = prepare_data(args, train=True)
    elif args.shift_type == 'label_shift_step':
        train_data, val_data = prepare_data(args, train=True)

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

    loader_kwargs = {'batch_size': args.batch_size, 'num_workers': 4, 'pin_memory': True}
    train_loader = train_data.get_loader(train=True, reweight_groups=args.reweight_groups, **loader_kwargs)
    val_loader = val_data.get_loader(train=False, reweight_groups=None, **loader_kwargs)
    if test_data is not None:
        test_loader = test_data.get_loader(train=False, reweight_groups=None, **loader_kwargs)

    data = {}
    data['train_loader'] = train_loader
    data['val_loader'] = val_loader
    data['test_loader'] = test_loader
    data['train_data'] = train_data
    data['val_data'] = val_data
    data['test_data'] = test_data
    n_classes = train_data.n_classes

    log_data(data, logger)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    ## Initialize model
    pretrained = not args.train_from_scratch
    if args.finetune_en == True:
        assert args.dataset == 'MultiNLI'

        from transformers import BertConfig

        config_class = BertConfig
        config = config_class.from_pretrained(
            'bert-base-uncased',
            num_labels=3,
            finetuning_task='mnli')
        model = BertForSequenceClassificationWithCovReg.from_pretrained(
            'bert-base-uncased',
            config=config,
            feature_size=args.features_size,
            device=device,
            reg_disentangle=args.reg_disentangle,
            reg_causal=args.reg_causal,
            disentangle_en=args.disentangle_en,
            counterfactual_en=args.counterfactual_en).to(device)
        print("Loading the disentangled model...")
        checkpoint = torch.load(os.path.join(args.load_model_path, 'best_model.pth'))
        model.load_state_dict(checkpoint.state_dict())
        print("Freeze all layers except the last layer")
        model = model_parameters_freeze(model)
        model.reg_disentangle = args.reg_disentangle
        model.reg_causal = args.reg_causal
        model.disentangle_en = args.disentangle_en
        model.counterfactual_en = args.counterfactual_en
    else:
        if resume:
            model = torch.load(os.path.join(args.log_dir, 'last_model.pth'))
            d = train_data.input_size()[0]
        elif model_attributes[args.model]['feature_type'] in ('precomputed', 'raw_flattened'):
            assert pretrained
            # Load precomputed features
            d = train_data.input_size()[0]
            model = nn.Linear(d, n_classes)
            model.has_aux_logits = False
        elif args.model == 'resnet50':
            model = torchvision.models.resnet50(pretrained=pretrained)
            d = model.fc.in_features
            model.fc = nn.Linear(d, n_classes)
        elif args.model == 'resnet34':
            model = torchvision.models.resnet34(pretrained=pretrained)
            d = model.fc.in_features
            model.fc = nn.Linear(d, n_classes)
        elif args.model == 'wideresnet50':
            model = torchvision.models.wide_resnet50_2(pretrained=pretrained)
            d = model.fc.in_features
            model.fc = nn.Linear(d, n_classes)
        elif args.model == 'bert':
            assert args.dataset == 'MultiNLI'

            from transformers import BertConfig, BertForSequenceClassification
            config_class = BertConfig
            model_class = BertForSequenceClassification

            config = config_class.from_pretrained(
                'bert-base-uncased',
                num_labels=3,
                finetuning_task='mnli')
            # model = model_class.from_pretrained(
            #     'bert-base-uncased',
            #     from_tf=False,
            #     config=config)
            model = BertForSequenceClassificationWithCovReg.from_pretrained(
                'bert-base-uncased',
                config=config,
                feature_size=args.features_size,
                device=device,
                reg_disentangle=args.reg_disentangle,
                reg_causal=args.reg_causal,
                disentangle_en=args.disentangle_en,
                counterfactual_en=args.counterfactual_en).to(device)
        else:
            raise ValueError('Model not recognized.')

    logger.flush()

    ## Define the objective
    if args.hinge:
        assert args.dataset in ['CelebA', 'CUB']  # Only supports binary

        def hinge_loss(yhat, y):
            # The torch loss takes in three arguments so we need to split yhat
            # It also expects classes in {+1.0, -1.0} whereas by default we give them in {0, 1}
            # Furthermore, if y = 1 it expects the first input to be higher instead of the second,
            # so we need to swap yhat[:, 0] and yhat[:, 1]...
            torch_loss = torch.nn.MarginRankingLoss(margin=1.0, reduction='none')
            y = (y.float() * 2.0) - 1.0
            return torch_loss(yhat[:, 1], yhat[:, 0], y)

        criterion = hinge_loss
    else:
        criterion = torch.nn.CrossEntropyLoss(reduction='none')

    if resume:
        df = pd.read_csv(os.path.join(args.log_dir, 'test.csv'))
        epoch_offset = df.loc[len(df) - 1, 'epoch'] + 1
        logger.write(f'starting from epoch {epoch_offset}')
    else:
        epoch_offset = 0
    train_csv_logger = CSVBatchLogger(os.path.join(args.log_dir, 'train.csv'), train_data.n_groups, mode=mode)
    val_csv_logger = CSVBatchLogger(os.path.join(args.log_dir, 'val.csv'), train_data.n_groups, mode=mode)
    test_csv_logger = CSVBatchLogger(os.path.join(args.log_dir, 'test.csv'), train_data.n_groups, mode=mode)

    train(model, criterion, data, logger, train_csv_logger, val_csv_logger, test_csv_logger, args,
          epoch_offset=epoch_offset)

    train_csv_logger.close()
    val_csv_logger.close()
    test_csv_logger.close()


def check_args(args):
    if args.shift_type == 'confounder':
        assert args.confounder_names
        assert args.target_name
    elif args.shift_type.startswith('label_shift'):
        assert args.minority_fraction
        assert args.imbalance_ratio


if __name__ == '__main__':
    main()
