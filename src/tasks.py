cancer = {
    'kostic_healthy_tumor_gg_otu': {
        'project': 'kostic',
        'type': 'classification',
        'multiple_samples': True,
        'data_type': 'OTU',
        'microbiome_file': 'gg/otutable.txt',
        'task_mapping_file': 'task.txt',
    },
    'kostic_healthy_tumor_refseq_otu': {
        'project': 'kostic',
        'type': 'classification',
        'multiple_samples': True,
        'data_type': 'OTU',
        'microbiome_file': 'refseq/otutable.txt',
        'task_mapping_file': 'task.txt',
    },
}

vaginal = {
    'ravel_white_black_gg_otu': {
        'project': 'ravel',
        'type': 'classification',
        'multiple_samples': False,
        'data_type': 'OTU',
        'microbiome_file': 'gg/otutable.txt',
        'task_mapping_file': 'task-white-black.txt',
    },
    'ravel_white_black_refseq_otu': {
        'project': 'ravel',
        'type': 'classification',
        'multiple_samples': False,
        'data_type': 'OTU',
        'microbiome_file': 'refseq/otutable.txt',
        'task_mapping_file': 'task-white-black.txt'
    },
    'ravel_nugent_category_gg_otu': {
        'project': 'ravel',
        'type': 'classification',
        'multiple_samples': False,
        'data_type': 'OTU',
        'microbiome_file': 'gg/otutable.txt',
        'task_mapping_file': 'task-nugent-category.txt',
    },
    'ravel_nugent_category_refseq_otu': {
        'project': 'ravel',
        'type': 'classification',
        'multiple_samples': False,
        'data_type': 'OTU',
        'microbiome_file': 'refseq/otutable.txt',
        'task_mapping_file': 'task-nugent-category.txt'
    }
}


cirrhosis = {
    'qin_healthy_cirrhosis_otu': {
        'project': 'qin2014',
        'type': 'classification',
        'multiple_samples': False,
        'data_type': 'OTU',
        'microbiome_file': 'otutable.txt',
        'task_mapping_file': 'task-healthy-cirrhosis.txt'
    },
}

