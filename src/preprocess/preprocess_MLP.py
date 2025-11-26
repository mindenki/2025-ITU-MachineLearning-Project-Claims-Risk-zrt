import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans


def preprocess(df: pd.DataFrame) -> ColumnTransformer:
    """
    Build a ColumnTransformer that:
    - drops 'Area' completely
    - treats 'VehPower' as ordinal (encoded + scaled)
    - scales numeric features
    - one-hot-encodes other categoricals
    """
    df = df.copy()
    if "Area" in df.columns:
        df_wo_area = df.drop(columns=["Area"])
    else:
        df_wo_area = df

    ord_cols = ["VehPower"]

    cat_cols = df_wo_area.select_dtypes(
        include=["object", "category", "string", "bool"]
    ).columns.tolist()

    if "VehPower" in cat_cols:
        cat_cols.remove("VehPower")

    num_cols = [c for c in df_wo_area.columns if c not in cat_cols + ord_cols]

    vehpower_cats = sorted(pd.Series(df["VehPower"]).unique().tolist())
    ord_cats = [vehpower_cats]

    ord_pipe = Pipeline([
        ("ord", OrdinalEncoder(
            categories=ord_cats,
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )),
        ("scale", StandardScaler()),
    ])

    ohe = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
        dtype=np.float32,
    )

    transformers = [
        ("num", StandardScaler(), num_cols),
        ("ord", ord_pipe, ord_cols),
        ("ohe", ohe, cat_cols),
    ]

    pre = ColumnTransformer(
        transformers,
        remainder="drop",
        sparse_threshold=0.0,
    )

    return pre


def make_pca(n_components: int, seed: int = 42) -> PCA:
    """Return a PCA model with given n_components."""
    return PCA(n_components=n_components, random_state=seed)


def prepare_datasets_with_clusters(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    n_components: int = 40,
    k_clusters: int = 5,
    seed: int = 42,
):
    """
    - drop ClaimRate into y
    - log1p + StandardScaler on y
    - preprocess X (drop Area, VehPower ordinal, others OHE)
    - PCA on preprocessed X (n_components)
    - StandardScaler on PCA outputs
    - KMeans (MiniBatch) on PCA space with k_clusters
    - append cluster label as extra feature
    Returns:
        X_train_final, X_test_final, y_train_std, y_test_std,
        dict of fitted objects for later use
    """
    target = "ClaimRate"

    X_train_raw = train_df.drop(columns=[target])
    X_test_raw  = test_df.drop(columns=[target])

    y_train = np.log1p(train_df[target].to_numpy()).astype(np.float32)
    y_test  = np.log1p(test_df[target].to_numpy()).astype(np.float32)

    y_scaler = StandardScaler().fit(y_train.reshape(-1, 1))
    y_train_std = y_scaler.transform(y_train.reshape(-1, 1)).ravel().astype(np.float32)
    y_test_std  = y_scaler.transform(y_test.reshape(-1, 1)).ravel().astype(np.float32)

    pre = preprocess(X_train_raw)
    X_train_pre = pre.fit_transform(X_train_raw)
    X_test_pre  = pre.transform(X_test_raw)

    pca_model = make_pca(n_components=n_components, seed=seed)
    X_train_pca = pca_model.fit_transform(X_train_pre).astype("float32")
    X_test_pca  = pca_model.transform(X_test_pre).astype("float32")

    x_scaler = StandardScaler().fit(X_train_pca)
    X_train_reduced = x_scaler.transform(X_train_pca).astype("float32")
    X_test_reduced  = x_scaler.transform(X_test_pca).astype("float32")

    kmeans = MiniBatchKMeans(
        n_clusters=k_clusters,
        init="k-means++",
        random_state=seed,
        batch_size=4096,
        max_iter=200,
        n_init="auto",
    )
    kmeans.fit(X_train_reduced)

    cluster_train = kmeans.predict(X_train_reduced).astype("int32")
    cluster_test  = kmeans.predict(X_test_reduced).astype("int32")

    X_train_final = np.column_stack([X_train_reduced, cluster_train])
    X_test_final  = np.column_stack([X_test_reduced,  cluster_test])

    artifacts = {
        "pre": pre,
        "pca": pca_model,
        "x_scaler": x_scaler,
        "kmeans": kmeans,
        "y_scaler": y_scaler,
    }

    return X_train_final, X_test_final, y_train_std, y_test_std, artifacts

