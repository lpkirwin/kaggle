"""Common modelling functions and objects.

Common stragies:

- GridSearchCV over an entire pipeline
- BayesSearchCV over an entire pipeline

"""

import pandas as pd
from sklearn.model_selection import cross_val_predict
from skopt import BayesSearchCV
from skopt.space import Categorical, Integer, Real
from os.path import join



# Temporary monkey patch
# https://github.com/scikit-optimize/scikit-optimize/issues/762
class BayesSearchCV2(BayesSearchCV):
    
    def __init__(self, estimator, search_spaces, optimizer_kwargs=None,
                 n_iter=50, scoring=None, fit_params=None, n_jobs=1,
                 n_points=1, iid=True, refit=True, cv=None, verbose=0,
                 pre_dispatch='2*n_jobs', random_state=None,
                 error_score='raise', return_train_score=False):

        self.search_spaces = search_spaces
        self.n_iter = n_iter
        self.n_points = n_points
        self.random_state = random_state
        self.optimizer_kwargs = optimizer_kwargs
        self._check_search_space(self.search_spaces)
        self.fit_params = fit_params

        super(BayesSearchCV, self).__init__(
            estimator=estimator, scoring=scoring,
            n_jobs=n_jobs, iid=iid, refit=refit, cv=cv, verbose=verbose,
            pre_dispatch=pre_dispatch, error_score=error_score,
            return_train_score=return_train_score)


PARAMS_SKOPT = {
    "lgb_small_trees": {
        "n_estimators": Integer(50, 2000),
        "max_depth": Integer(1, 8),
        "num_leaves": Integer(4, 32),
        "learning_rate": Real(0.0001, 10, prior="log-uniform"),
        "cat_smooth": Real(0.01, 100, prior="log-uniform"),
    },
    "lgb_big_trees": {
        "n_estimators": Integer(5, 500),
        "max_depth": Integer(100, 200),
        "num_leaves": Integer(50, 500),
        "learning_rate": Real(0.0001, 10, prior="log-uniform"),
        "cat_smooth": Real(0.01, 100, prior="log-uniform"),
    },
}

# # Allegedly good params from kaggle post for santander
# {'bagging_fraction': 0.8360034886892089,
#  'feature_fraction': 0.1,
#  'learning_rate': 0.09520535659597275,
#  'max_bin': 20,
#  'max_depth': 23,
#  'min_data_in_leaf': 20,
#  'min_sum_hessian_in_leaf': 100.0,
#  'num_leaves': 80,
#  'subsample': 0.03708289604738438,
#  'objective': 'binary',
#  'metric': 'auc',
#  'is_unbalance': True,
#  'boost_from_average': False}

PARAMS_GRID = {}

def get_oof_predictions(estimator, X, y, cv=5, fit_params=None, n_jobs=-1):
    return cross_val_predict(estimator, X, y, cv=cv, n_jobs=n_jobs, fit_params=fit_params)


def run_estimator_cv(estimator_cv, fit_params, X, y):
    """E.g. either GridSearchCV or BayesSearchCV"""
    estimator_cv.fit(X, y, **fit_params)
    print("Best score:", estimator_cv.best_score_)
    best_params = estimator_cv.best_params_
    best_est = estimator_cv.best_estimator_
    cv_df = pd.DataFrame(estimator_cv.cv_results_)
    return best_params, best_est, cv_df
