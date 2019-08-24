"""Common modelling functions and objects.

Common stragies:

- GridSearchCV over an entire pipeline
- BayesSearchCV over an entire pipeline

"""

import pandas as pd
from sklearn.model_selection import cross_val_predict
from skopt import BayesSearchCV
from os.path import join


# Temporary monkey patch (until skopt updates to match sklearn)
# https://github.com/scikit-optimize/scikit-optimize/issues/762
class BayesSearchCV2(BayesSearchCV):
    def __init__(
        self,
        estimator,
        search_spaces,
        optimizer_kwargs=None,
        n_iter=50,
        scoring=None,
        fit_params=None,
        n_jobs=1,
        n_points=1,
        iid=True,
        refit=True,
        cv=None,
        verbose=0,
        pre_dispatch="2*n_jobs",
        random_state=None,
        error_score="raise",
        return_train_score=False,
    ):

        self.search_spaces = search_spaces
        self.n_iter = n_iter
        self.n_points = n_points
        self.random_state = random_state
        self.optimizer_kwargs = optimizer_kwargs
        self._check_search_space(self.search_spaces)
        self.fit_params = fit_params

        super(BayesSearchCV, self).__init__(
            estimator=estimator,
            scoring=scoring,
            n_jobs=n_jobs,
            iid=iid,
            refit=refit,
            cv=cv,
            verbose=verbose,
            pre_dispatch=pre_dispatch,
            error_score=error_score,
            return_train_score=return_train_score,
        )


def get_oof_predictions(estimator, X, y, cv=5, fit_params=None, n_jobs=-1):
    return cross_val_predict(
        estimator, X, y, cv=cv, n_jobs=n_jobs, fit_params=fit_params
    )


def run_estimator_cv(estimator_cv, fit_params, X, y):
    """E.g. either GridSearchCV or BayesSearchCV"""
    estimator_cv.fit(X, y, **fit_params)
    print("Best score:", estimator_cv.best_score_)
    best_params = estimator_cv.best_params_
    best_est = estimator_cv.best_estimator_
    cv_df = pd.DataFrame(estimator_cv.cv_results_)
    return best_params, best_est, cv_df
