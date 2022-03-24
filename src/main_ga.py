import dataset
import config
from ga import GA

import random
import string
import datetime
import os

import pandas as pd
import numpy as np
import ast

from pathlib import Path


def run(data, output_path):
    #
    # Open stats files
    #
    f_kfold = open(str(output_path) + '/best_kfold.csv', "w")
    f_kfold.write('search_id;num_genes;metric;mo_fitness;mask;features\n')
    f_test = open(str(output_path) + '/best_test.csv', "w")
    f_test.write('search_id;num_genes;metric;mo_fitness;mask;features\n')

    #
    # Update GA parameters with dataset dependency
    #
    config.GA_PARAMS['gen_size'] = data.num_features - 1
    config.GA_PARAMS['mutate_init_prob'] = config.GA_PARAMS['init_ind_length']/data.num_features
    config.GA_PARAMS['mutate_dup'] = 1/data.num_features

    #
    # Run GA search n times
    #
    for search_id in range(config.GA_PARAMS['n_searches']):

        print('\n\n====================================================================')
        print('GA search:', search_id)

        ga = GA(config.GA_PARAMS, data, output_path, search_id)
        ga.search()
        data.create_folds()

        #
        # Write stats
        #
        for item in ga.stats['kfold']:
            f_kfold.write('{};{};{};{};{};{}\n'.format(
                item[0], item[1], item[2], item[3], item[4], item[5]
            ))

        for item in ga.stats['test']:
            f_test.write('{};{};{};{};{};{}\n'.format(
                item[0], item[1], item[2], item[3], item[4], item[5]
            ))

    f_kfold.close()
    f_test.close()


def print_results(output_path):
    test_results = pd.read_csv(str(output_path) + '/best_test.csv', sep=';')

    df_max = test_results.groupby('search_id').agg({'mo_fitness': ['max']})
    df_max.columns = df_max.columns.map(lambda x: '|'.join([str(i) for i in x]))

    df_merged = pd.merge(test_results, df_max, left_on=['search_id', 'mo_fitness'], right_on=['search_id', 'mo_fitness|max']).sort_values('search_id', ascending=True)
    df_merged.drop(['mo_fitness|max'], axis=1, inplace=True)

    print('\n\nPrinting GA search results:')

    for row in df_merged.itertuples():
        print(f"\nSearch:       {row[1]}")
        print(f"Metric:       {np.round(row[3], 3)}")
        print(f"Num Features: {row[2]}")

        features = ast.literal_eval(row[6])
        print("Features:")
        for feature in features:
            print(f"    {feature}")


if __name__ == "__main__":

    #
    # Create folder for log files
    #
    ga_id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
    timestamp   = datetime.datetime.now().strftime("%Y%m%d_%H_%M_%S")

    path = Path(config.DATA_PARAMS['path'])
    output_path = Path(config.DATA_PARAMS['path'] + '/output/{}_{}_{}'.format(ga_id, config.DATA_PARAMS['task']['project'], timestamp))
    print('\nPreparing result folder:', output_path)
    os.mkdir(output_path)

    #
    # Build dataset including preprocessing
    #
    print('\nBuilding dataset.')
    data = dataset.MicrobiomeDataset(**config.DATA_PARAMS)
    data.create_folds()


    run(data, output_path)

    print_results(output_path)
