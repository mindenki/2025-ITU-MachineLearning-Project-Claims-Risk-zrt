import pandas as pd 
import numpy as np
import os
from pathlib import Path
from scipy import sparse

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"



def load_data(raw: bool=True, target: str=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads the raw data from the data directory as pandas DataFrames.
    If raw is False, loads the processed data, as numpy arrays and sparse matrices.
    Parameters:
    ----------
    data_dir : str
        The directory where the data is stored.
    raw : bool
        If True, loads the raw data. If False, loads the processed data.
    target : str, optional
        The target variable to load.
    Returns:
    -------
    train_df : pd.DataFrame
        The training data
    test_df : pd.DataFrame
        The test data
    """
    if raw:
        train_path = RAW_DIR / "claims_raw_train.csv"
        test_path  = RAW_DIR / "claims_raw_test.csv"
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Training data not found at {train_path}")
        if not os.path.exists(test_path):
            raise FileNotFoundError(f"Test data not found at {test_path}")
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        return train_df, test_df
        
    else:
        X_train_path = PROC_DIR / "X_train.npz"
        X_test_path = PROC_DIR / "X_test.npz"
        y_train_path = PROC_DIR / f"y_train_{target}.npy"
        y_test_path = PROC_DIR / f"y_test_{target}.npy"
        if not os.path.exists(X_train_path):
            raise FileNotFoundError(f"Training data not found at {X_train_path}")
        if not os.path.exists(X_test_path):
            raise FileNotFoundError(f"Test data not found at {X_test_path}")
        if not os.path.exists(y_test_path):
            raise FileNotFoundError(f"Test data not found at {y_test_path}")
        if not os.path.exists(y_train_path):
            raise FileNotFoundError(f"Training data not found at {y_train_path}")
        
        X_train = sparse.load_npz(X_train_path)
        X_test = sparse.load_npz(X_test_path)
        y_train = np.load(y_train_path)
        y_test = np.load(y_test_path)
        
        return (X_train, y_train), (X_test, y_test)
        
    
    

if  __name__ == "__main__":
    train_df, test_df = load_data(raw=True)
    print("Raw Data")
    print(train_df.head())
    