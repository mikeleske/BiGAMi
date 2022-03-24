import pandas as pd
import numpy as np
import copy
from pathlib import Path
from collections import defaultdict
from sklearn.model_selection import KFold, StratifiedKFold

import preprocessing


class MicrobiomeDataset:

    def __init__(
        self,
        path,
        task,
        folds=None,
        scale=False,
        transformation=None,
        selectKBest=None
    ):
        """
        :document
        """
        self.path = path
        self.task = task
        self.folds  = folds
        self.scale = scale
        self.transformation = transformation
        self.selectKBest = selectKBest

        self.features = None
        self.num_features = None

        self.df, self.df_orig = self.read_data()

    def __len__(self):
        return self.df.shape[1]

    def read_data(self):

        data_file = self.task['microbiome_file']
        task_file = self.task['task_mapping_file']

        data_folder = Path(self.path + '/input/{}/'.format(self.task['project']))
        df = pd.read_csv(Path(str(data_folder) + '/' + data_file), sep='\t', index_col=0)

        task = pd.read_csv(Path(str(data_folder) + '/' + task_file), sep='\t')

        self.features = df.index.values
        self.num_features = df.shape[0]


        #
        # Transpose data
        #
        df = preprocessing.transpose_dataframe(df)


        #
        # Merge data with target and control
        #
        df['#SampleID'] = df['#SampleID'].astype(str)
        task['#SampleID'] = task['#SampleID'].astype(str)
        
        df = preprocessing.merge_df_task(df, task)


        #
        # Encode classification label
        #
        if self.task['type'] == 'classification':
            df = preprocessing.encode_label(df)


        #
        # Store copy of untransformed data
        #
        df_orig = copy.deepcopy(df)


        #
        # Drop Zero Variance Features, i.e. 'all-zeros'
        #
        df, self.features, self.num_features = preprocessing.dropVariance0(df, self.features)


        #
        # Transform data to Relative abundances or CLR
        #
        if self.transformation == 'rel':
            df = preprocessing.transform_relative_abundance(df, self.num_features)
        elif self.transformation == 'clr':
            df = preprocessing.transform_composition(df, self.num_features, self.transformation)


        #
        # Scale data
        #
        if self.scale:
            df = preprocessing.minmax_scale(df, self.features)


        #
        # Reduce to k best features
        #
        if self.selectKBest:
           df, self.features, self.num_features = preprocessing.selectKbest(df, self.features, max_features=self.selectKBest)


        print('Final dataset shape:', df.shape)

        print()


        return df, df_orig


    def create_folds(self):

        if self.task['multiple_samples']:
            ctrl_vars = np.unique(self.df.ControlVar)
            np.random.shuffle(ctrl_vars)

            mapping = defaultdict(int)

            for fold, sub in enumerate(np.array_split(ctrl_vars, self.folds)):
                for ctrl_var in sub:
                    mapping[ctrl_var] = fold

            self.df['kfold'] = self.df.ControlVar.map(mapping)

        else:

            # we create a new column called kfold and fill it with -1
            self.df["kfold"] = -1

            # the next step is to randomize the rows of the data
            self.df = self.df.sample(frac=1).reset_index(drop=True)

            # fetch labels
            y = self.df.Var.values

            kf = None
            if self.task['type'] == 'regression':
                kf = KFold(n_splits=self.folds)
            elif self.task['type'] == 'classification':
                kf = StratifiedKFold(n_splits=self.folds)

            # fill the new kfold column
            for f, (t_, v_) in enumerate(kf.split(X=self.df, y=y)):
                self.df.loc[v_, 'kfold'] = f


    #def set_folds(self, folds):
    #    self.df['kfold'] = folds
