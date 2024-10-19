from train import *

def train_nlp():
    args = SimpleNamespace(
        dataset_name=DATASET_NAME, #'yelp-author-style', #'MultiNLI',  # 'civilcomments',
        disentangle_version=int(F + 1),
        cpns_version=int(C * (D + 1) + E),
        reweight_version=int(A * (B + 1)),
        n_exp=0,
        model_name='bert-base-uncased',

        lr=3e-1,
        n_epochs=15,
        batch_size=32,
        feature_size=128,

        finetune_flg=True,
        reweight_flg=True,  # True

        weight_decay=1e-4,
        reg_disentangle=0.5,
        reg_causal=2,
        gamma_reweight=0.001,
        dfr_reweighting_frac=0,  # 0.2,
    )
    if args.dataset_name == 'civilcomments':
        train_civil(args)
    elif args.dataset_name == 'MultiNLI':
        train_nli(args)
    elif args.dataset_name == 'yelp-author-style' or 'beer-concept-occurrence':
        train_yelp(args)


if __name__ == '__main__':
    train_nlp()
