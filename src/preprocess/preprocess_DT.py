# src/data/preprocess_dt.py

from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


def preprocess_dt(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str = "ClaimRate",
):

    y_train = train_df[target_col].to_numpy()
    y_test = test_df[target_col].to_numpy()

    X_train = train_df.drop(columns=[target_col]).copy()
    X_test = test_df.drop(columns=[target_col]).copy()

    cat_cols = X_train.select_dtypes(
        include=["object", "category", "string", "bool"]
    ).columns.tolist()
    num_cols = [c for c in X_train.columns if c not in cat_cols]
    ordinal_cols = ["VehPower"]
    # Area is redundant, density can replace it
    # ordinal_cols = []
    # if "Area" in cat_cols:
    #     ordinal_cols.append("Area")
    #     cat_cols.remove("Area")

    num_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
        ]
    )

    cat_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                    dtype=np.float32,
                ),
            ),
        ]
    )

    transformers = []
    if num_cols:
        transformers.append(("num", num_pipeline, num_cols))

    if cat_cols:
        transformers.append(("cat", cat_pipeline, cat_cols))

    if ordinal_cols:
        VehPower_order = sorted(
            X_train["VehPower"].dropna().unique().tolist()
        )

        ordinal_pipeline = Pipeline(
            steps=[
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("ord", OrdinalEncoder(categories=[VehPower_order])),
            ]
        )
        transformers.append(("ord", ordinal_pipeline, ordinal_cols))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.3,
    )

    X_train_pre = preprocessor.fit_transform(X_train)
    X_test_pre = preprocessor.transform(X_test)

    return X_train_pre, y_train, X_test_pre, y_test
