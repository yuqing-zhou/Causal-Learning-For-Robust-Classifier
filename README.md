# Fighting Spurious Correlations in Text Classification via a Causal Learning Perspective

This is the repository for the paper "Fighting Spurious Correlations in Text Classification via a Causal Learning Perspective." 

### File Structure
To successfully run the program, the project file should follow the followiing structure:
```
`
├──code
│   ├──data
│   ├──gdro_fork
│   ├──losses
│   ├──optimizers
│   ├──utils
│   ├──wilds_exps
│   ├──wilds_exps_utils
│   ├──main.py
│   ├──model.py
│   ├──myutils.py
│   ├──train.py
│   ├──utils_glue.py
│   ├──run.slurm
│   └──README.md
└──data
│   ├──datasets
│   │   ├──beer-concept-occurrence
│   │   ├──civilcomments_v1.0
│   │   ├──multinli_bert_features
│   │   └──yelp-author-style
│   └──models
└──README.md
```
### Datasets

CivilComments: We use the version of the dataset available in the [WILDS package](https://wilds.stanford.edu/datasets/#civilcomments).

MultiNLI:  This dataset can be downloaded [here](https://nlp.stanford.edu/data/dro/multinli_bert_features.tar.gz).

beer-concept-occurrence: We upload the data in data/dataset. The original datasets can be downloaded [here](https://github.com/yuqing-zhou/shortcut-learning-in-text-classification/tree/master/Dataset/beer_new/target/concept/split/occurrence/occurrence2). We select 2 ratings (0.6 and 1.0) and replace their labels with 0 and 1, respectively. 

yelp-author-style: We upload the data in data/dataset. The original dataset can be downloaded [here](https://github.com/yuqing-zhou/shortcut-learning-in-text-classification/tree/master/Dataset/yelp_review_full_csv/style/split/author/author1). We select 2 ratings (2 and 4) and replace their labels with 0 and 1, respectively. 


### Run
To start the program, run main.py.

Modify the DATASET_NAME.

For the first stage, set finetune_flg=False and reweight_flg=False.

For the second stage, set finetune_flg=True and reweight_flg=True.
