import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Literal, List
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder, FunctionTransformer
from sklearn.decomposition import TruncatedSVD




# Manual Preprocessing

def preprocess_manual(df: pd.DataFrame, exposure: bool = False) -> pd.DataFrame:
    """
    Some basic preprocessing steps applied manually to the dataframe. These are not part of the pipeline and just some basic cleaning steps.
    """
    
    # Region: group rare regions into "Other"
    if "Region" in df.columns:
        freq = df["Region"].value_counts(normalize=True)
        df["Region"] = df["Region"].apply(lambda x: x if freq[x] > 0.01 else "Other")
    
    
    # Exposure 
    if "Exposure" in df.columns:
        df["Exposure"] = df["Exposure"].clip(lower=0.01)
        df["Exposure"] = df["Exposure"].clip(upper=1.0)
    
    # Also remove unnecessary columns if present
    unnecessary_cols = ["IDpol", "ClaimNb", "ClaimRate", "log_ClaimRate", "Area", "Exposure"]
    if exposure:
        unnecessary_cols.remove("Exposure")
    for col in unnecessary_cols:
        if col in df.columns:
            df = df.drop(columns=[col])
    
    return df



# create a dataclass to hold preprocessing configuration

@dataclass
class FeatureConfig:
    name: str
    feature_type: str # "numerical", "categorical", "ordinal" etc.
    bins: List[float] | None = None
    scale: bool = True
    #imput_strategy: str = "median" # not really needed but could be useful when data is missing a lot of values
    
    
    
FEATURES= [
    FeatureConfig(name="Exposure", feature_type="numerical"),
    FeatureConfig(name="VehPower", feature_type="ordinal", bins=[4,5,6,7,8,10,12,100]),
    #FeatureConfig(name="VehAge", feature_type="numerical"),
    FeatureConfig(name="VehAge", feature_type="ordinal", bins=[0,1,3,5,10,15,20,30,100]),
    FeatureConfig(name="DrivAge", feature_type="numerical"),
    FeatureConfig(name="Density", feature_type="log"),
    FeatureConfig(name="BonusMalus", feature_type="numerical"),
    FeatureConfig(name="VehBrand", feature_type="categorical"),
    FeatureConfig(name="VehGas", feature_type="categorical"),
    FeatureConfig(name="Region", feature_type="categorical"),
]



# Custom Transformers

class Binner:
    def __init__(self, bins):
        self.bins = bins

    def __call__(self, X):
        X = pd.DataFrame(X)
        out = X.apply(lambda col: pd.cut(col, bins=self.bins, labels=False))
        return out.values

def make_binner(bins: list) -> FunctionTransformer:
    return FunctionTransformer(Binner(bins), feature_names_out="one-to-one")


class LogTransform:
    def __call__(self, X):
        return np.log1p(X)

def make_log_transformer():
    return FunctionTransformer(LogTransform(), feature_names_out="one-to-one")


# def make_capper(cap: float = .99) -> FunctionTransformer:
#     def capper(X):
#         X = np.asarray(X)
#         upper_cap = np.quantile(X, cap, axis=0)
#         return np.minimum(X, upper_cap)
#     return FunctionTransformer(capper, feature_names_out="one-to-one")

# def make_ratio_transformer():
#     def ratio(X):
#         return X[:, 0] / X[:, 1].reshape(-1, 1)
#     def ratio_name(transformer, feature_names_in):
#         return ["ratio"]
    
#     return FunctionTransformer(ratio, feature_names_out=ratio_name)


# Definition of reusable pipeline components


def make_numeric_pipeline(scale: bool = True) -> Pipeline:
    steps = [
        #("imputer", SimpleImputer(strategy=imput_strategy))
    ]
    if scale:
        steps.append(("scaler", StandardScaler()))
    return Pipeline(steps)

def make_log_pipeline(imput_strategy: str = "median") -> Pipeline:
    return Pipeline([
        #("imputer", SimpleImputer(strategy=imput_strategy)),
        ("log_transform", make_log_transformer()),
        ("scaler", StandardScaler())
    ])

def make_categorical_pipeline(imput_strategy: str = "most_frequent") -> Pipeline:
    return Pipeline([
        #("imputer", SimpleImputer(strategy=imput_strategy)),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

def make_ordinal_pipeline(imput_strategy: str = "most_frequent") -> Pipeline:
    return Pipeline([
        #("imputer", SimpleImputer(strategy=imput_strategy)),
        ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
    ])


        

# Building the full ColumnTransformer pipeline

def build_feature_pipeline(feature_configs: List[FeatureConfig] = FEATURES) -> ColumnTransformer:
    transformers = []
    
    for config in feature_configs:
        if config.feature_type == "numerical":
            transformers.append((config.name, make_numeric_pipeline(config.scale), [config.name]))
        elif config.feature_type == "log":
            transformers.append((config.name, make_log_pipeline(), [config.name]))
        elif config.feature_type == "categorical":
            transformers.append((config.name, make_categorical_pipeline(), [config.name]))
        elif config.feature_type == "ordinal":
            if config.bins:
                binner = make_binner(config.bins)
                ordinal_pipeline = make_pipeline(
                    #SimpleImputer(strategy=config.imput_strategy),
                    binner,
                    OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
                )
                transformers.append((config.name, ordinal_pipeline, [config.name]))
            else:
                transformers.append((config.name, make_ordinal_pipeline(), [config.name]))
        else:
            raise ValueError(f"Unknown feature type: {config.feature_type}")
    
    
    return ColumnTransformer(transformers)


# Target creation

TargetType = Literal["ClaimNb", "ClaimRate", "log_ClaimRate"]

def create_target(df: pd.DataFrame, target: TargetType, claimnbcol: str = "ClaimNb",
                  exposurecol: str = "Exposure", exposure_floor: float = 0.01,
                  add_to_df: bool = False, new_col_name: str | None = None) -> pd.Series:
    if target not in TargetType.__args__:
        raise ValueError(f"{target} type has not been implemented. Must be one of {TargetType.__args__}")
    
    exposure = df[exposurecol].clip(upper=1.0)
    
    if target == "ClaimNb": 
        target_series = df[claimnbcol]
        default_new_col_name = "ClaimNb"
    elif target == "ClaimRate":
        target_series = df[claimnbcol] / exposure
        default_new_col_name = "ClaimRate"
    elif target == "log_ClaimRate":
        target_series = np.log(df[claimnbcol] / exposure.clip(lower=exposure_floor) + 1)
        default_new_col_name = "log_ClaimRate"
        
    # more target types can be added here in the future if needed
    
    
    if new_col_name is None:
        new_col_name = default_new_col_name
        
    if add_to_df:
        df[new_col_name] = target_series
        
    return target_series


if __name__ == "__main__":
    pass