# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.0.2
#   kernelspec:
#     display_name: kaggle
#     language: python
#     name: kaggle
# ---

# %%
import pandas as pd
import pandas_profiling as pp

# %%
# %load_ext autoreload
# %autoreload 2

# %%
from src.util import set_context, raw_path, comp_path, reduce_mem_usage

# %%
set_context("titanic")

# %%
tr = pd.read_csv(raw_path("train.csv"))
te = pd.read_csv(raw_path("test.csv"))
tr.shape, te.shape

# %%
col_diff = set(tr.columns).difference(te.columns)
assert len(col_diff) == 1
target_col = col_diff.pop()
target_col

# %%
tr["_test"] = False
te["_test"] = True

# %%
df = pd.concat([tr, te], sort=True)
df.shape

# %%
df.sample(10).sort_values("_test")

# %%
df.info(memory_usage="deep")

# %%
df = reduce_mem_usage(df)

# %%
obj_cols = df.select_dtypes("object").columns.tolist()
for col in obj_cols:
    print(f"{col}: {df[col].nunique()} unique values")
    df[col] = df[col].astype("category")

# %%
df.describe()

# %%
# pp.ProfileReport(df)

# %%
df.to_pickle(comp_path("clean.pkl"))
