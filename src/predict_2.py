import sys
from pathlib import Path
from typing import Union, Optional

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = PROJECT_ROOT / "models" / "pipeline_self_mlr.joblib"


def _load_joblib(path: Union[str, Path]):
    p = Path(path)
    if p.exists():
        return joblib.load(p)
    return None


def _prepare_input(input_data, model=None) -> pd.DataFrame:
    """Accept several input types and return a DataFrame ready for prediction.

    Supported types: CSV path, dict (single row), list[dict], DataFrame, Series, numpy array.
    If a numpy array is provided and the loaded model exposes `feature_names_in_`,
    those names are used as column names.
    """
    if isinstance(input_data, (str, Path)):
        p = Path(input_data)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {p}")
        return pd.read_csv(p)

    if isinstance(input_data, dict):
        return pd.DataFrame([input_data])

    if isinstance(input_data, list):
        return pd.DataFrame(input_data)

    if isinstance(input_data, pd.Series):
        return input_data.to_frame().T

    if isinstance(input_data, pd.DataFrame):
        return input_data.copy()

    if isinstance(input_data, np.ndarray):
        arr = input_data
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if model is not None and hasattr(model, "feature_names_in_"):
            cols = list(model.feature_names_in_)
            return pd.DataFrame(arr, columns=cols)
        cols = [f"f{i}" for i in range(arr.shape[1])]
        return pd.DataFrame(arr, columns=cols)

    raise TypeError(f"Unsupported input type: {type(input_data)}")


def _ensure_nine_features(X: pd.DataFrame) -> pd.DataFrame:
    """Ensure DataFrame has the 9 features expected by the original pipeline.

    Accepts inputs with 6 features: `age, sex, bmi, children, smoker, region`
    and expands to: age, sex, bmi, children, smoker, region_northeast, region_northwest,
    region_southeast, region_southwest following the transformations in `predict.py`.
    """
    X = X.copy()

    # Drop common target column if present
    for target_col in ("charges", "target", "y"):
        if target_col in X.columns:
            X = X.drop(columns=[target_col])

    # If already has region_* columns and encoded sex/smoker, assume ready
    needed_regions = [
        "region_northeast", "region_northwest", "region_southeast", "region_southwest"
    ]
    if set(needed_regions).issubset(set(X.columns)) and all(c in X.columns for c in ["age", "bmi", "children", "sex", "smoker"]):
        return X

    # Map sex: original script uses 0 for female, 1 for male
    if "sex" in X.columns:
        if X["sex"].dtype == object:
            X["sex"] = X["sex"].str.lower().map(lambda v: 0 if v == "female" else 1)
        else:
            X["sex"] = X["sex"].astype(int)

    # Map smoker: original script mapped 'yes' -> 0, 'no' -> 1
    if "smoker" in X.columns:
        if X["smoker"].dtype == object:
            X["smoker"] = X["smoker"].str.lower().map(lambda v: 0 if v == "yes" else 1)
        else:
            X["smoker"] = X["smoker"].astype(int)

    # One-hot region
    if "region" in X.columns:
        regions = ["northeast", "northwest", "southeast", "southwest"]
        for r in regions:
            col = f"region_{r}"
            X[col] = 0
        # set whichever region present
        X["region"] = X["region"].str.lower()
        for r in regions:
            mask = X["region"] == r
            X.loc[mask, f"region_{r}"] = 1
        X = X.drop(columns=["region"])

    # Ensure region columns exist even if region not provided
    for col in needed_regions:
        if col not in X.columns:
            X[col] = 0

    return X


EXPECTED_FEATURES = [
    "age", "sex", "bmi", "children", "smoker",
    "region_northeast", "region_northwest", "region_southeast", "region_southwest"
]


def predict(input_data,
            model_path: Union[str, Path] = DEFAULT_MODEL) -> pd.Series:
    """Return predictions for the provided input.

    Parameters
    - input_data: dict | list[dict] | DataFrame | Series | numpy array | CSV path
    - model_path: path to joblib pipeline
    - scaler_path: optional path to a scaler joblib. If not provided and
      `models/scaler.joblib` exists it will be used.

    Returns
    - pandas.Series of predictions (index matches input rows)
    """
    model = _load_joblib(model_path)
    if model is None:
        raise FileNotFoundError(f"Model not found at {model_path}")

    X = _prepare_input(input_data, model=model)
    # Accept 6-feature inputs (age, sex, bmi, children, smoker, region)
    # and expand to the 9-feature format expected by the original script.
    X = _ensure_nine_features(X)

    # Ensure columns match expected features (9-feature model)
    for col in EXPECTED_FEATURES:
        if col not in X.columns:
            X[col] = 0
    X = X[EXPECTED_FEATURES]

    preds = model.predict(X)
    return pd.Series(preds, index=X.index, name="prediction")


def _cli():
    if len(sys.argv) < 2:
        print("Usage: python src/predict_2.py <input.csv>\nOr pass a CSV path as the only argument.")
        return
    inp = sys.argv[1]
    preds = predict(inp)
    # Prepare output file with timestamp and model name
    from datetime import datetime
    in_df = _prepare_input(inp)
    # If input was provided without the 9 features, show the transformed input used
    out_df = _ensure_nine_features(in_df)
    out_df = out_df.copy()
    out_df["prediction"] = preds.values

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = Path(DEFAULT_MODEL).stem
    out_dir = PROJECT_ROOT / "predictions"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{ts}_{model_name}.csv"
    out_df.to_csv(out_path, index=False)

    # Print predictions and saved path
    print(out_df.to_csv(index=False))
    print(f"Saved predictions to: {out_path}")


if __name__ == "__main__":
    _cli()
