from tasks import cancer, vaginal, cirrhosis

DATA_PARAMS = {
    'path': r'/Users/mleske/Documents/dev/microbiome-git',

    #'task': cancer['kostic_healthy_tumor_gg_otu'],
    'task': vaginal['ravel_nugent_category_refseq_otu'],
    #'task': vaginal['ravel_white_black_refseq_otu'],
    #'task': cirrhosis['qin_healthy_cirrhosis_otu'],

    'folds': 6,
    'scale': True,
    'transformation': 'clr',        # 'raw', 'rel', 'clr'
    'selectKBest': 128
}

GA_PARAMS = {
    'n_searches': 25,
    'pop_size': 250,
    'max_iter': 10,
    'weights': (1.0, -1.0),
    'init': 'zero',                 # zero, random, full
    'init_ind_length': 10,
    'penalty_denominator': 250,
    'gen_size': None,
    'mutate_init_prob': None,
    'bestN': 1,
    'crossover': '1p',
    'CXPB': 0.8,
    'MUTPB': 0.8
}

TRAIN_PARAMS = {
    'type': 'classification',
    'metric': 'auc',                # 'acc', 'auc'
    'alg': 'SGDClassifier',
    'fi': 'coef',
}
