# src/data_loader.py
from pathlib import Path
import pandas as pd
import joblib
from typing import Tuple

ROOT = Path(__file__).resolve().parents[1]  # project root
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

# class Data_Loader:
    # def __init__(self, data_dir:) -> None:



def read_csv(filename: str = "insurance.csv") -> pd.DataFrame:
    path = DATA_DIR / filename
    df = pd.read_csv(path)
    return df

def save_artifact(obj, name: str):
    path = MODELS_DIR / name
    joblib.dump(obj, path)
    return str(path)

def load_artifact(name: str):
    path = MODELS_DIR / name
    return joblib.load(path)

def save_splits(x_train, x_test, y_train, y_test):
    joblib.dump(x_train, MODELS_DIR / "x_train.joblib")
    joblib.dump(x_test, MODELS_DIR / "x_test.joblib")
    joblib.dump(y_train, MODELS_DIR / "y_train.joblib")
    joblib.dump(y_test, MODELS_DIR / "y_test.joblib")
    return True

def load_splits() -> Tuple:
    x_train = joblib.load(MODELS_DIR / "x_train.joblib")
    x_test = joblib.load(MODELS_DIR / "x_test.joblib")
    y_train = joblib.load(MODELS_DIR / "y_train.joblib")
    y_test = joblib.load(MODELS_DIR / "y_test.joblib")
    return x_train, x_test, y_train, y_test
