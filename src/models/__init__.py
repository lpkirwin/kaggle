"""Common modelling functions and objects.

Common stragies:

- GridSearchCV over an entire pipeline
- BayesSearchCV over an entire pipeline

"""

from skopt.space import Real, Categorical, Integer

from sklearn.model_selection import cross_val_predict

import pandas as pd

PARAMS_SKOPT = {
    "lgb_small_trees": {
        "num_iterations": Integer(100, 2000),
        "max_depth": Integer(1, 8),
        "num_leaves": Integer(4, 16),
        "learning_rate": Real(0.0001, 10, prior="log-uniform"),
    },
    "lgb_big_trees": {
        "num_iterations": Integer(5, 500),
        "max_depth": Integer(100, 200),
        "num_leaves": Integer(100, 500),
        "learning_rate": Real(0.0001, 10, prior="log-uniform")
    }
}

PARAMS_GRID = {

}

def get_oof_predictions(estimator, fit_params, cv, X, y, n_jobs=-1):

    cross_val_predict(estimator, X, y, cv=cv, n_jobs=n_jobs, fit_params=fit_params)

def run_estimator_cv(estimator_cv, fit_params, X, y):
    """E.g. either GridSearchCV or BayesSearchCV"""
    estimator_cv.fit(X, y, **fit_params)
    print("Best score:", estimator_cv.best_score_)
    best_params = estimator_cv.best_params_
    best_est = estimator_cv.best_estimator_
    cv_df = pd.DataFrame(estimator_cv.cv_results_)
    return best_params, best_est, cv_df