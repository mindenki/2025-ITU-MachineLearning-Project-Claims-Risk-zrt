import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.decomposition import TruncatedSVD

def preprocess(df: pd.DataFrame) -> ColumnTransformer: 
    ord_cols = ["Area", "VehPower"]
    cat_cols = df.select_dtypes(include=["object", "category", "string", "bool"]).columns.tolist()
    num_cols = [c for c in df.columns if c not in cat_cols and c not in ord_cols]

    cat_cols.remove("Area")
    ord_cats = []
    ord_cats.append( sorted(pd.Series(df["Area"]).unique().tolist()))
    ord_cats.append( sorted(pd.Series(df["VehPower"]).unique().tolist()))

    
    transformers = []
    transformers.append(("num", StandardScaler(), num_cols))
    transformers.append(("ord", OrdinalEncoder(categories=ord_cats, handle_unknown="use_encoded_value", unknown_value=-1), ord_cols))
    transformers.append(("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float32), cat_cols))
    
    pre = ColumnTransformer(transformers, remainder="drop")
    
    return pre

# Example usage
# pre = preprocess(df)
# train = pre.fit_transform(train_df)
# test = pre.transform(test_df)

def PCA(n_components: int, seed: int = 42) -> TruncatedSVD:
    return TruncatedSVD(n_components=n_components, algorithm="randomized", n_iter=7, random_state=seed)

# Example usage
# pca = PCA(X.shape[1], seed=seed)
# X_train_reduced = pca.fit_transform(X_train)
# X_test_reduced = pca.transform(X_test)