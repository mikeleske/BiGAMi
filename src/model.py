from sklearn.linear_model import LogisticRegression, Lasso, LinearRegression, SGDRegressor
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

classifiers = {
    'BaggingClassifier': BaggingClassifier(),
    'DecisionTreeClassifier': DecisionTreeClassifier(),
    'RandomForestClassifier': RandomForestClassifier(n_estimators=20),
    'AdaBoostClassifier': AdaBoostClassifier(n_estimators=20, random_state=1),
    'GradientBoostingClassifier': GradientBoostingClassifier(n_estimators=20, random_state=1),
    'SGDClassifier': SGDClassifier(random_state=1, loss="log", penalty="l1", max_iter=500),
    'LogisticRegression': LogisticRegression(random_state=1, penalty='l1', solver='liblinear', max_iter=5000),
    'Lasso': Lasso(random_state=1),
}

regressors = {
    'LinearRegression': LinearRegression(),
    'SGDRegressor': SGDRegressor(penalty="l1", max_iter=20000),
    'DecisionTreeRegressor': DecisionTreeRegressor(),
    'Lasso': Lasso(random_state=1),
}
