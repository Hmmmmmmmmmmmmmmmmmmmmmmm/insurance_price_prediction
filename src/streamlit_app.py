import sys
from pathlib import Path
import io
from datetime import datetime

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src import evaluate as eval_mod
from src.predict import predict as predict_single
from src.predict_2 import predict as predict_batch

MODEL_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
PRED_DIR = PROJECT_ROOT / "predictions"
PRED_DIR.mkdir(exist_ok=True)

st.set_page_config(page_title="Insurance Models Dashboard", layout="wide")

st.title("Insurance Price Prediction — Models Dashboard")

# Sidebar: choose model
st.sidebar.header("Model / Data")

# Discover available model files (only include the three canonical pipelines)
canonical = {
    "pipeline_lr": "pipeline_lr.joblib",
    "pipeline_self_mlr": "pipeline_self_mlr.joblib",
    "pipe_poly": "pipe_poly.joblib",
}

available = {name: MODEL_DIR / fname for name, fname in canonical.items() if (MODEL_DIR / fname).exists()}

if not available:
    st.sidebar.error("No supported models found in models/. Place pipeline_lr.joblib, pipeline_self_mlr.joblib or pipe_poly.joblib in the models folder.")
    model_choice = None
else:
    model_choice = st.sidebar.selectbox("Select model:", list(available.keys()))

# Resolve model_path (may be None if no models found)
model_path = available.get(model_choice) if model_choice else None

# Load model and data when requested
@st.cache_resource
def load_model(p):
    return joblib.load(p)

@st.cache_data
def load_data(name):
    p = PROJECT_ROOT / "models" / f"{name}.joblib"
    if p.exists():
        return joblib.load(p)
    # Fallback: try loading sample data
    try:
        return pd.read_csv(PROJECT_ROOT / "data" / "insurance.csv")
    except Exception:
        return None

model = None
try:
    model = load_model(model_path)
except Exception as e:
    st.sidebar.error(f"Failed to load model: {e}")

# Load train/test artifacts if present
x_train = load_data("x_train")
y_train = load_data("y_train")
x_test = load_data("x_test")
y_test = load_data("y_test")

# Main: show dashboard
st.header("Model Evaluation")
if model is None:
    st.warning("No model loaded — check the models folder.")
else:
    st.markdown(f"**Using model:** {model_path.name}")
    if isinstance(x_train, pd.DataFrame) and isinstance(y_train, (pd.Series, list, tuple)) and isinstance(x_test, pd.DataFrame) and isinstance(y_test, (pd.Series, list, tuple)):
        with st.spinner("Rendering dashboard..."):
            fig = eval_mod.dashboard_model(model, x_train, y_train, x_test, y_test, model_name=model_path.stem, partial_dependence=False, normalize_coefs=True, return_figure=True)
            if fig is not None:
                st.pyplot(fig)
            else:
                st.write("Dashboard could not be produced.")
    else:
        st.write("Train/test artifacts not available — cannot render dashboard.\nYou can still run predictions below.")

st.markdown("---")
st.header("Predict")
st.write("Choose single input or upload a CSV of inputs. Single input uses `src/predict.py` logic; CSV uses `src/predict_2.py`.")
mode = st.radio("Input mode:", ("Single", "CSV Upload"))

if mode == "Single":
    st.subheader("Manual input")
    col1, col2, col3 = st.columns(3)
    age = col1.number_input("age", value=35, min_value=0)
    sex = col1.selectbox("sex", ("male", "female"))
    bmi = col2.number_input("bmi", value=27.5, format="%f")
    children = col2.number_input("children", value=0, min_value=0)
    smoker = col3.selectbox("smoker", ("yes", "no"))
    region = col3.selectbox("region", ("northeast", "northwest", "southeast", "southwest"))

    if st.button("Predict (single)"):
        # Build dict and encode as predict.py expects
        ui = {
            "age": int(age),
            "sex": sex,
            "bmi": float(bmi),
            "children": int(children),
            "smoker": smoker,
            "region": region,
        }
        # Apply same transforms as original predict.py
        ui_local = ui.copy()
        ui_local["smoker"] = 0 if ui_local["smoker"] == "yes" else 1
        ui_local["sex"] = 0 if ui_local["sex"] == "female" else 1
        region_val = ui_local.pop("region")
        ui_local["region_northeast"] = 0
        ui_local["region_northwest"] = 0
        ui_local["region_southeast"] = 0
        ui_local["region_southwest"] = 0
        if region_val == "northeast":
            ui_local["region_northeast"] = 1
        elif region_val == "northwest":
            ui_local["region_northwest"] = 1
        elif region_val == "southeast":
            ui_local["region_southeast"] = 1
        elif region_val == "southwest":
            ui_local["region_southwest"] = 1

        try:
            pred = predict_single(ui_local)
            st.success(f"Predicted insurance cost: ${pred:,.2f}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

else:
    st.subheader("Upload CSV")
    uploaded = st.file_uploader("Upload a CSV with columns: age,sex,bmi,children,smoker,region (or full 9-feature set)", type=["csv"])
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
        except Exception:
            st.error("Could not read the uploaded CSV — ensure it's a valid CSV")
            df = None

        if df is not None:
            st.write("Preview of input:")
            st.dataframe(df.head())
            if st.button("Predict (CSV)"):
                try:
                    preds = predict_batch(df)
                    out_df = df.copy()
                    # If input didn't include region dummies, predict_batch will expand internally; align rows
                    out_df = out_df.reset_index(drop=True)
                    out_df["prediction"] = preds.values

                    csv_bytes = out_df.to_csv(index=False).encode("utf-8")
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    model_name = model_path.stem if model_path is not None else "model"
                    out_name = f"{ts}_{model_name}.csv"
                    out_path = PRED_DIR / out_name
                    out_df.to_csv(out_path, index=False)

                    st.success(f"Predictions saved to {out_path}")
                    st.download_button("Download predictions CSV", data=csv_bytes, file_name=out_name, mime="text/csv")
                    st.dataframe(out_df.head())
                except Exception as e:
                    st.error(f"Batch prediction failed: {e}")

st.markdown("---")
st.write("Notes: Single prediction uses `src/predict.py` internals; CSV uses `src/predict_2.py` which will also save results to `predictions/`.")
