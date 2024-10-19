DATASET_INFO = {
    'civilcomments': {
        'num_classes': 2,
        'dataset_path': '',
        'transform': None,
    },
    'MultiNLI': {
        'num_classes': 3,
        'dataset_path': 'multinli_bert_features/',
        'transform': None,
    },
    'waterbirds': {
        'num_classes': 2,
        'dataset_path': 'datasets/waterbird_complete95_forest2water2/',
        'transform': 'AugWaterbirdsCelebATransform',
    },
    'celeba': {
        'num_classes': 2,
        'dataset_path': 'datasets/celebA/',
        'transform': 'AugWaterbirdsCelebATransform',
    },
    'yelp-author-style': {
        'num_classes': 2,
        'dataset_path': 'yelp-author-style',
        'transform': None,
    },
    'beer-concept-occurrence': {
        'num_classes': 2,
        'dataset_path': 'beer-concept-occurrence',
        'transform': None,
    },
    'beer-corr': {
        'num_classes': 2,
        'dataset_path': 'beer-corr',
        'transform': None,
    },
}