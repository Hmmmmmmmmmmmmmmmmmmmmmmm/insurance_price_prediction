# Insurance Price Prediction

Small project exhibiting the limitataion of a Linear Model using Insurance Price Prediction.

### Files of interest:
-`notebooks/evaluation.ipynb` for detailed evaluation of all 3 models
- `src/predict.py` - interactive single-row predictor (existing).
- `src/predict_2.py` - flexible batch predictor that accepts CSV, DataFrame, dict, list-of-dicts, numpy arrays, or pandas Series. It also looks for an external scaler at `models/scaler.joblib` and applies it if the pipeline does not already include scaling.
- `models/` - contains saved pipelines (e.g. `pipeline_self_mlr.joblib`).


### File Structure:

INSURANCE_PRICE_PREDICTION/
├── data/
│ ├── insurance_no_outliers.csv
│ └── insurance.csv
│
├── models/
│ ├── pipe_poly.joblib
│ ├── pipeline_lr.joblib
│ ├── pipeline_self_mlr.joblib
│ ├── x_test.joblib
│ ├── x_train.joblib
│ ├── y_test.joblib
│ └── y_train.joblib
│
├── notebooks/
│ ├── 01_data_understanding_and_EDA.ipynb
│ ├── 02_modeling.ipynb
│ └── 03_evaluation.ipynb
│
├── predictions/
│
├── src/
│
├── .gitignore
├── README_streamlit.md
├── README.md
└── requirements.txt

### Details:

This project contains 3 regression models:
1. Linear Regression with OLS
2. Linear Regression using Gradient Decent
3. Linear Regression with Polynomial Features

The project show how Accuracy of each models differs despite using cleaned, appropriate and similar dataset being used and highlights the limitation of a Linear Regression model being used to predict a using multiple features. This project is mainly made to exhibit academic understanding of Regression and highlighting it's limitation.

### Usage examples:

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
