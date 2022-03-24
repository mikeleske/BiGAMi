import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, chi2, VarianceThreshold
from skbio.stats.composition import clr


def transpose_dataframe(df):
    samples = df.columns
    df = pd.DataFrame(data=df.values.T, index=None, columns=df.index).rename_axis(None, axis=1)
    df['#SampleID'] = samples
    return df

def encode_label(df):
    le = LabelEncoder()
    le.fit(df['Var'])
    df['Var'] = le.transform(df['Var'])
    return df

def merge_df_task(df, task):
    return pd.merge(df, task, on='#SampleID', how='inner')

def transform_composition(df, num_features, alg):
    if alg == 'clr':
        df.iloc[:, :num_features] = clr(df.values[:, :num_features].astype(float) + 0.5)
    return df

def transform_relative_abundance(df, num_features):
    df_values = df.iloc[:, :num_features].values
    df.iloc[:, :num_features] = df_values/(df_values.sum(axis=1, keepdims=True) + 0.00001)
    return df

def minmax_scale(df, features):
    result = df.copy()
    
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(result[features].values)
    result.iloc[:, :scaled_data.shape[1]] = scaled_data

    return result

def dropVariance0(df, df_features):
    df_new = df.copy()

    features = []
    mask = [ i for i in range(0, len(df_features)) ]

    selector = VarianceThreshold().fit(df_new.iloc[:, mask], df_new.Var)
    features += [ f[0] for f in zip(mask, selector.get_support()) if f[1] ]

    filter_list = []
    for f in zip(mask, selector.get_support()):
        if not f[1]: filter_list.append(f[0])

    features = df_features[features]
    slice_columns = list(features) + ['#SampleID'] + list(df_new.columns[len(df_features)+1:])

    return df[slice_columns], features, len(features)

def selectKbest(df, df_features, max_features=512):

    df_new = df.copy()

    features = []
    mask = [ i for i in range(0, len(df_features)) ]

    selector = SelectKBest(chi2, k=max_features).fit(df_new.iloc[:, mask], df_new.Var)
    features += [ f[0] for f in zip(mask, selector.get_support()) if f[1] ]

    features = df_features[features]
    slice_columns = list(features) + ['#SampleID'] + list(df_new.columns[len(df_features)+1:])

    return df[slice_columns], features, len(features)