import pandas as pd 
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
DATA_DIR = os.path.join(ROOT_DIR, "data")



def load_data(data_dir: str="../data", raw: bool=True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads the raw data from the data directory.
    Parameters:
    ----------
    data_dir : str
        The directory where the data is stored.
    raw : bool
        If True, loads the raw data. If False, loads the processed data.
    Returns:
    -------
    train_df : pd.DataFrame
        The training data
    test_df : pd.DataFrame
        The test data
    """
    if raw:
        train_path = os.path.join(data_dir, "raw", "claims_raw_train.csv")
        test_path = os.path.join(data_dir, "raw", "claims_raw_test.csv")
    else:
        train_path = os.path.join(data_dir, "processed", "claims_processed_train.csv")
        test_path = os.path.join(data_dir, "processed", "claims_processed_test.csv")
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training data not found at {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test data not found at {test_path}")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df

if  __name__ == "__main__":
    train_df, test_df = load_data(raw=True, data_dir=DATA_DIR)
    print("Raw Data")
    print(train_df.head())
    