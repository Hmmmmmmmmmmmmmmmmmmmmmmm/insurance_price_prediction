Streamlit UI — Insurance Models Dashboard

This README explains the Streamlit-based UI added to the project.

Files
- `src/streamlit_app.py`: Streamlit application. Provides:
  - Evaluation dashboard per model (uses `src/evaluate.dashboard_model` and local train/test artifacts when available).
  - Single-row prediction UI that uses the existing `src/predict.py` logic.
  - CSV upload for batch predictions using `src/predict_2.py` (results saved into `predictions/` and downloadable).

Run the app

From the project root, install requirements and run:

```bash
pip install -r requirements.txt
streamlit run src/streamlit_app.py
```

Notes
- The app will attempt to load models from `models/` and train/test artifacts `models/x_train.joblib`, `models/y_train.joblib`, `models/x_test.joblib`, `models/y_test.joblib` to render dashboards.
- If artifacts are missing, the dashboard area will explain that the train/test data are unavailable, but prediction features still work.
- Single prediction uses the same feature-encoding conventions as `src/predict.py` (sex: female->0, male->1; smoker: yes->0, no->1; region expanded into four region_* columns).

Security
- Do not deploy this app publicly without appropriate input validation and access controls. Uploaded CSVs are processed on the server.
