"""Skopt parameter spaces for different estimators."""

from skopt.space import Categorical, Integer, Real

lgb_small_trees = {
    "n_estimators": Integer(50, 2000),
    "max_depth": Integer(1, 8),
    "num_leaves": Integer(4, 32),
    "learning_rate": Real(0.0001, 10, prior="log-uniform"),
    "cat_smooth": Real(0.01, 100, prior="log-uniform"),
}

lgb_big_trees = {
    "n_estimators": Integer(5, 500),
    "max_depth": Integer(100, 200),
    "num_leaves": Integer(50, 500),
    "learning_rate": Real(0.0001, 10, prior="log-uniform"),
    "cat_smooth": Real(0.01, 100, prior="log-uniform"),
}


# Params from kaggle kernel
lgb_santander_params = {
    "bagging_fraction": 0.8360034886892089,
    "feature_fraction": 0.1,
    "learning_rate": 0.09520535659597275,
    "max_bin": 20,
    "max_depth": 23,
    "min_data_in_leaf": 20,
    "min_sum_hessian_in_leaf": 100.0,
    "num_leaves": 80,
    "subsample": 0.03708289604738438,
    "objective": "binary",
    "metric": "auc",
    "is_unbalance": True,
    "boost_from_average": False,
}

