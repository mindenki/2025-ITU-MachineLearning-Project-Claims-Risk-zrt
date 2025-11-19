import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.data.load_data import load_data
from src.data.preprocess_dt import preprocess_dt
from src.models.dt import DT


def evaluate_regression(y_true, y_pred, split_name: str) -> None:
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"===== {split_name} =====")
    print(f"MSE : {mse:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE : {mae:.6f}")
    print(f"R^2 : {r2:.6f}")
    print()


train_df, test_df = load_data(raw=False)

X_train, y_train, X_test, y_test = preprocess_dt(train_df, test_df)

model = DT(
    criterion="squared_error",   # or "poisson"
    max_depth=10,
    min_samples_split=20,
    min_samples_leaf=50,
    random_state=42,
)

model.fit(X_train, y_train)

y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

evaluate_regression(y_train, y_pred_train, split_name="TRAIN")
evaluate_regression(y_test, y_pred_test, split_name="TEST")