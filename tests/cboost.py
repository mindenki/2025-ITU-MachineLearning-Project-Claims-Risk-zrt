import sys, os, numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
from src.data.load_data import load_data
from sklearn.model_selection import train_test_split

SEED = 42
np.random.seed(SEED)

train_df, test_df = load_data(raw=False)

X_train = train_df.drop(columns=["ClaimRate"])
X_test  = test_df.drop(columns=["ClaimRate"])
y_train = np.log1p(train_df["ClaimRate"]).astype(np.float32)
y_test_z = np.log1p(test_df["ClaimRate"]).astype(np.float32)


cat_features = X_train.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
cat_features.append("VehPower")

param_grid = [
    dict(depth=6, learning_rate=0.05, l2_leaf_reg=10, bagging_temperature=1.0),
    dict(depth=8, learning_rate=0.05, l2_leaf_reg=10, bagging_temperature=1.0),
    dict(depth=8, learning_rate=0.03, l2_leaf_reg=20, bagging_temperature=0.5),
    dict(depth=10, learning_rate=0.05, l2_leaf_reg=10, bagging_temperature=1.0),
]

def run_cv(params):
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    oof_z = np.zeros(len(X_train), dtype=np.float32)
    val_scores = []

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train), 1):
        X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
        y_tr, y_va = y_train[tr_idx], y_train[va_idx]

        train_pool = Pool(X_tr, y_tr, cat_features=cat_features)
        valid_pool = Pool(X_va, y_va, cat_features=cat_features)

        model = CatBoostRegressor(
            loss_function="RMSE",
            eval_metric="RMSE",
            iterations=8000,
            early_stopping_rounds=300,
            random_seed=SEED,
            verbose=False,
            **params,
        )
        model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
        z_pred = model.predict(valid_pool)
        oof_z[va_idx] = z_pred.astype(np.float32)

        rmse_z = mean_squared_error(y_va, z_pred)
        val_scores.append(rmse_z)
        print(f"[Fold {fold}] RMSE(log1p rate)={rmse_z:.6f}")

    mean_rmse = float(np.mean(val_scores))
    print(f"[CV] Params={params} -> mean RMSE(log1p rate)={mean_rmse:.6f}")
    return mean_rmse, oof_z

best = (1e9, None, None)
for params in param_grid:
    score, oof_z = run_cv(params)
    if score < best[0]:
        best = (score, params, oof_z)

best_score, best_params, best_oof_z = best
print("\n=== Best CV config ===")
print(best_params)
print(f"Best mean RMSE(log1p rate): {best_score:.6f}")

oof_rate = np.expm1(best_oof_z)
true_rate = train_df["ClaimRate"].to_numpy()
oof_rmse = mean_squared_error(true_rate, oof_rate)
oof_mae  = mean_absolute_error(true_rate, oof_rate)
print(f"OOF RMSE(rate)={oof_rmse:.6f}, OOF MAE(rate)={oof_mae:.6f}")

def poisson_deviance(y, mu):
    eps = 1e-12
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    term = np.where(y > 0, y * np.log((y + eps) / (mu + eps)) - (y - mu), -mu)
    return 2.0 * np.sum(term) / max(1, y.shape[0])

if {"ClaimNb", "Exposure"}.issubset(train_df.columns):
    mu_oof = np.clip(oof_rate * train_df["Exposure"].to_numpy(), 1e-12, None)
    pd_oof = poisson_deviance(train_df["ClaimNb"].to_numpy(), mu_oof)
    print(f"OOF Poisson deviance (counts)={pd_oof:.6f}")


X_tr, X_va, y_tr, y_va = train_test_split(
    X_train, y_train, test_size=0.15, random_state=SEED, shuffle=True
)
train_pool = Pool(X_tr, y_tr, cat_features=cat_features)
valid_pool = Pool(X_va, y_va, cat_features=cat_features)
test_pool  = Pool(X_test, y_test_z, cat_features=cat_features)

final_model = CatBoostRegressor(
    loss_function="RMSE",
    eval_metric="RMSE",
    iterations=8000,
    early_stopping_rounds=300,
    random_seed=SEED,
    verbose=200,
    **best_params,
)
final_model.fit(train_pool, eval_set=valid_pool, use_best_model=True)

z_test = final_model.predict(test_pool)
rate_test = np.expm1(z_test)
rate_true_test = test_df["ClaimRate"].to_numpy()

rmse_test = mean_squared_error(rate_true_test, rate_test)
mae_test  = mean_absolute_error(rate_true_test, rate_test)
print("\n=== Test Evaluation (rate) ===")
print(f"RMSE={rmse_test:.6f}, MAE={mae_test:.6f}")

if {"ClaimNb", "Exposure"}.issubset(test_df.columns):
    mu_test = np.clip(rate_test * test_df["Exposure"].to_numpy(), 1e-12, None)
    pd_test = poisson_deviance(test_df["ClaimNb"].to_numpy(), mu_test)
    print(f"Poisson deviance (counts)={pd_test:.6f}")