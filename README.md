# Insurance Price Prediction

Small project showing models and pipelines to predict insurance charges.

Files of interest:
- `src/predict.py` - interactive single-row predictor (existing).
- `src/predict_2.py` - flexible batch predictor that accepts CSV, DataFrame, dict, list-of-dicts, numpy arrays, or pandas Series. It also looks for an external scaler at `models/scaler.joblib` and applies it if the pipeline does not already include scaling.
- `models/` - contains saved pipelines (e.g. `pipeline_self_mlr.joblib`).

Usage examples

- Predict from a CSV and print CSV of predictions:
```bash
python src/predict_2.py data/input.csv
```

- Use `predict()` from Python:
```python
from src.predict_2 import predict
import pandas as pd

df = pd.read_csv('data/insurance_no_outliers.csv')
preds = predict(df)
print(preds.head())
```

Notes
- `predict_2.py` will try to load `models/pipeline_self_mlr.joblib` by default. Pass a different model path to `predict()` if needed.
- If an external scaler is saved at `models/scaler.joblib`, the script will use it unless the loaded pipeline already contains a scaler step.

Requirements
- See `requirements.txt` for the minimal dependencies.
