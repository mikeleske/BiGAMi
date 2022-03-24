from sklearn import metrics

def train(df, control, model, metric, fold, test_fold, mask=None):
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    # Remove the hold out test set (last fold)
    df_train = df_train[df_train.kfold != test_fold]

    drop_cols = []
    if control:
        drop_cols = ['#SampleID', 'ControlVar', 'Var', 'kfold']
    else:
        drop_cols = ['#SampleID', 'Var', 'kfold']

    x_train = df_train.drop(drop_cols, axis=1).values
    y_train = df_train.Var.values

    x_valid = df_valid.drop(drop_cols, axis=1).values
    y_valid = df_valid.Var.values

    if mask:
        x_train = x_train[:, mask]
        x_valid = x_valid[:, mask]

    model.fit(x_train, y_train)
    preds = model.predict(x_valid)

    score = None
    if metric == 'acc':
        score = metrics.accuracy_score(y_valid, preds)
    elif metric == 'auc':
        preds_proba = model.predict_proba(x_valid)[:, 1]
        score = metrics.roc_auc_score(y_valid, preds_proba)

    return score
