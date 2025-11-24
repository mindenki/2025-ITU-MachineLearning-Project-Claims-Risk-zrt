import pandas as pd
import numpy as np
from typing import Literal
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder, FunctionTransformer
from sklearn.decomposition import TruncatedSVD




# Custom Transformers


def make_log_transformer() -> FunctionTransformer:
    def log_transformer(X):
        return np.log1p(X)
    return FunctionTransformer(log_transformer, feature_names_out="one-to-one")

def make_binner(bins: list) -> FunctionTransformer:
    def binner(X):
        X = pd.DataFrame(X)
        out = X.apply(lambda col: pd.cut(col, bins= bins, labels= False))
        return out.values
    return FunctionTransformer(binner, feature_names_out="one-to-one")

def make_capper(cap: float = .99) -> FunctionTransformer:
    def capper(X):
        X = np.asarray(X)
        upper_cap = np.quantile(X, cap, axis=0)
        return np.minimum(X, upper_cap)
    return FunctionTransformer(capper, feature_names_out="one-to-one")

def make_ratio_transformer():
    def ratio(X):
        return X[:, 0] / X[:, 1].reshape(-1, 1)
    def ratio_name(transformer, feature_names_in):
        return ["ratio"]
    
    return FunctionTransformer(ratio, feature_names_out=ratio_name)

        

# Prepocessing Piplein Builder

def building_pipeline(*, 
                      log_features= None,
                      bin_features= None,
                      bin_specs=None,
                      cap_features=None,
                      cap_q: float = 0.99,
                      ratio_pairs=None,
                      categorical_features=None,
                      ordinal_features=None,
                      numeric_default_features=None
                      ) -> ColumnTransformer:
    
    transformers = []
    
    
    # Log Transformations
    if log_features:
        log_pipeline= make_pipeline(
            SimpleImputer(strategy="median"),
            make_log_transformer(),
            StandardScaler()
        )
        transformers.append(("log", log_pipeline, log_features))
    
    
    # Binning Transformations
    if bin_specs:
        for feature, bins in bin_specs.items():
            pipeline_name = f"bin_{feature}"
            binner = make_binner(bins)
            bin_pipeline = make_pipeline(
                SimpleImputer(strategy="median"),
                binner
            )
            transformers.append((pipeline_name, bin_pipeline, [feature]))
    
    
    # Capping Transformations
    if cap_features:
        cap_pipeline = make_pipeline(
            SimpleImputer(strategy="median"),
            make_capper(q=cap_q),
            StandardScaler()
        )
        transformers.append(("cap", cap_pipeline, cap_features))


    # Ratio Transformations
    if ratio_pairs:
        for (a, b) in ratio_pairs:
            ratio_pipeline = make_pipeline(
                SimpleImputer(strategy="median"),
                make_ratio_transformer(),
                StandardScaler()
            )
            transformers.append((f"ratio_{a}_div_{b}", ratio_pipeline, [a, b]))






    # Categorical Transformations
    if categorical_features:
        cat_pipeline = make_pipeline(
            SimpleImputer(strategy="most_frequent"),
            OneHotEncoder(handle_unknown="ignore")
        )
        transformers.append(("cat", cat_pipeline, categorical_features))
    
    
    
    # Ordinal Features
    if ordinal_features:
        ord_pipeline = make_pipeline(
            SimpleImputer(strategy="most_frequent"),
            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        )
        transformers.append(("ord", ord_pipeline, ordinal_features))
    
    
    # Numeric Default Features
    default_num_pipeline = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler()
    )

    return ColumnTransformer(
        transformers,
        remainder=default_num_pipeline
    )












TargetType = Literal["ClaimNb", "ClaimRate", "log_ClaimRate"]

def create_target(df: pd.DataFrame, target: TargetType, claimnbcol: str = "ClaimNb",
                  exposurecol: str = "Exposure", exposure_floor: float = 0.01,
                  add_to_df: bool = False, new_col_name: str | None = None) -> pd.Series:
    if target not in TargetType.__args__:
        raise ValueError(f"{target} type has not been implemented. Must be one of {TargetType.__args__}")
    
    if target == "ClaimNb": 
        target_series = df[claimnbcol]
        default_new_col_name = "ClaimNb"
    elif target == "ClaimRate":
        target_series = df[claimnbcol] / df[exposurecol]
        default_new_col_name = "ClaimRate"
    elif target == "log_ClaimRate":
        target_series = np.log(df[claimnbcol] / df[exposurecol].clip(lower=exposure_floor) + 1)
        default_new_col_name = "log_ClaimRate"
        
    # more target types can be added here in the future if needed
    
    
    if new_col_name is None:
        new_col_name = default_new_col_name
        
    if add_to_df:
        df[new_col_name] = target_series
        
    return target_series



# def preprocess(df: pd.DataFrame) -> ColumnTransformer: 
#     ord_cols = ["Area", "VehPower"]
#     cat_cols = df.select_dtypes(include=["object", "category", "string", "bool"]).columns.tolist()
#     num_cols = [c for c in df.columns if c not in cat_cols and c not in ord_cols]

#     cat_cols.remove("Area")
#     ord_cats = []
#     ord_cats.append( sorted(pd.Series(df["Area"]).unique().tolist()))
#     ord_cats.append( sorted(pd.Series(df["VehPower"]).unique().tolist()))

    
#     transformers = []
#     transformers.append(("num", StandardScaler(), num_cols))
#     transformers.append(("ord", OrdinalEncoder(categories=ord_cats, handle_unknown="use_encoded_value", unknown_value=-1), ord_cols))
#     transformers.append(("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float32), cat_cols))
    
#     pre = ColumnTransformer(transformers, remainder="drop")
    
#     return pre

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

if __name__ == "__main__":
    pass