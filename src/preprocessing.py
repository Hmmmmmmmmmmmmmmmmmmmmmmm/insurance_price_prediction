from enum import unique
import pandas as pd
import numpy as np
# from data_loader import load_raw, load_clean
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

CATEGORICAL_COLS = ["sex", "smoker", "region"]
NUMERICAL_COLS = ["age", "bmi", "children"]

def one_hot_encode(df)-> pd.DataFrame:
    encoder = OneHotEncoder()
    encoded = encoder.fit_transform(df[['region']]).toarray()
    print(type(encoded))
    encoder_df = pd.DataFrame(
        encoded, columns=encoder.get_feature_names_out()
    )
    df=pd.concat([df,encoder_df],axis=1)
    # df.head()
    return df


def split_features(df, target_col):
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    return X, y

def clean_data(df):
    df = df.copy()
    df = df.dropna()
    return df

# def get_processed_data(use_outlier_free=True):
#     df = load_clean() if use_outlier_free else load_raw()
#     df = clean_data(df)
#     df = one_hot_encode(df)
#     return split_features(df)

def transform_input_dict(input_dict):
    df = pd.DataFrame([input_dict])
    return one_hot_encode(df)

def outlier_cleaning(target_col, df):
    # Select only feature columns for cleaning
    feature_cols = df.drop(columns=target_col)
    feature_cols = feature_cols.select_dtypes(include="number")

    # Compute 5-number summary for all features at once
    summary = feature_cols.describe().loc[['min', '25%', '50%', '75%', 'max']]
    print("5-number summary for features:")
    print(summary, "\n")

    # Compute IQR for all features
    Q1 = feature_cols.quantile(0.25)
    Q3 = feature_cols.quantile(0.75)
    IQR = Q3 - Q1

    # Compute lower and upper fences
    lower_fence = Q1 - 1.5 * IQR
    upper_fence = Q3 + 1.5 * IQR

    # Create mask for all feature columns at once
    mask = feature_cols.apply(lambda x: x.between(lower_fence[x.name], upper_fence[x.name]))

    # Combine mask across columns: keep rows where all features are within IQR range
    mask_all = mask.all(axis=1)

    # Apply mask to original dataframe (target stays intact)
    df_cleaned = df[mask_all]
    # Rows that were removed (outliers)
    df_dropped = df[~mask_all]  # ~ inverts the boolean mask

    print("Dropped rows (outliers):")
    print(df_dropped)
    df_dropped.shape
    return df_dropped,df_cleaned



def split(X,y,test_size=0.20,random_state=42):
    x_train, x_test, y_train, y_test = train_test_split(
                                        X,y,
                                        test_size=test_size,
                                        random_state=random_state)
    return x_train, x_test, y_train, y_test


def binary_mapping(df, cols)-> pd.DataFrame:
    for col in cols:
        uniques = df[col].dropna().unique()
        if len(uniques) != 2:
            raise ValueError(f"Column {col} does not have exactly 2 unique values")
        mapping = {uniques[0]: 0, uniques[1]: 1}
        df.loc[:, col] = df[col].map(mapping)
    return df


def preprocessing_pipeline_optimal(df, binary_cols, target_col, encoding_cols):
    df=binary_mapping(df,binary_cols)
    df=one_hot_encode(df,encoding_cols)
    for col in encoding_cols:
        df = df.drop(col, axis=1)


def preprocessing_for_pipeline(df: pd.DataFrame, target_col: str | None = None, return_target: bool = False):
        """
        Recommended preprocessing function for preparing raw input for the project's
        pipelines. This function is intended as a single canonical transformer you can
        call before fitting or predicting with any of the three saved pipelines:

            - `pipeline_lr`       : sklearn Pipeline([StandardScaler(), LinearRegression()])
            - `pipeline_self_lr`  : Pipeline([StandardScalerSelf(), Linear_Regression_Self()])
            - `pipe_poly`         : Pipeline([PolynomialFeatures(...), StandardScaler(), LinearRegression()])

        Purpose
        -------
        - Accepts raw DataFrame (the raw CSV from `data/`) and returns a features
            DataFrame with the exact numeric columns that the models expect.
        - Encodes `sex` and `smoker` to binary integers using the same mapping the
            interactive `predict.py` uses (sex: female->0, male->1; smoker: yes->0, no->1).
        - One-hot encodes `region` into four indicator columns:
            `region_northeast`, `region_northwest`, `region_southeast`, `region_southwest`.
        - Drops a typical target column if present (e.g., `charges`) and optionally
            returns it.

        Usage Notes for each pipeline
        -----------------------------
        - `pipeline_lr` and `pipeline_self_lr`:
                * These pipelines include a scaler as the first step, so do NOT scale
                    numeric columns before passing the DataFrame to the pipeline. Pass the
                    DataFrame returned by this function directly to `.fit()` or `.predict()`.

        - `pipe_poly`:
                * This pipeline begins with `PolynomialFeatures`. It expects numeric
                    inputs (no object dtype). The output of this function is numeric and
                    ordered, so it can be passed directly to `pipe_poly`.
                * Do not apply `PolynomialFeatures` outside the saved pipeline; let the
                    saved pipeline handle it so the feature ordering is consistent.

        Parameters
        ----------
        df : pd.DataFrame
                Raw input data. Accepts DataFrames that contain the raw columns: at a
                minimum the function expects `age, sex, bmi, children, smoker, region`.
                If region is already one-hot encoded and `sex`/`smoker` already numeric,
                the function is idempotent and will return a copy with ensured ordering.

        target_col : str | None
                If provided and present in `df`, the column will be separated from the
                features. If `return_target` is True the target is returned as a second
                value.

        return_target : bool
                If True and `target_col` is provided, return `(X, y)`. Otherwise return
                `X` alone.

        Returns
        -------
        X : pd.DataFrame
                Features DataFrame with these columns, in this order:
                `age, sex, bmi, children, smoker, region_northeast, region_northwest,
                 region_southeast, region_southwest`

        y : pd.Series (optional)
                Target series if `return_target` is True and `target_col` supplied.

        Implementation details
        ----------------------
        - Mapping conventions mirror the existing interactive `predict.py`:
                sex: 'female' -> 0, 'male' -> 1
                smoker: 'yes' -> 0, 'no' -> 1
        - Region values are lower-cased before one-hot encoding so inputs like
            'SouthWest' or 'southwest' are treated equivalently.
        - Missing region values will result in all-zero region_* indicators.

        Note
        ----
        This function is provided as a canonical preprocessing helper. Do NOT call
        it from other library code automatically; instead import and call it from
        training or prediction scripts when you explicitly want to prepare raw data
        prior to passing it to a pipeline.
        """
        df_proc = df.copy()

        # Drop rows with NA (simple and conservative); caller may perform more
        # advanced imputation if desired.
        df_proc = df_proc.dropna()

        # Separate target if requested
        y = None
        if target_col and target_col in df_proc.columns:
                y = df_proc[target_col].copy()
                df_proc = df_proc.drop(columns=[target_col])

        # Ensure numeric columns exist
        for col in ["age", "bmi", "children"]:
                if col in df_proc.columns:
                        df_proc[col] = pd.to_numeric(df_proc[col], errors="coerce")

        # Binary mappings
        if "sex" in df_proc.columns:
                df_proc["sex"] = df_proc["sex"].astype(str).str.lower().map(lambda v: 0 if v == "female" else 1)
        if "smoker" in df_proc.columns:
                df_proc["smoker"] = df_proc["smoker"].astype(str).str.lower().map(lambda v: 0 if v == "yes" else 1)

        # One-hot encode region into four columns; keep deterministic column names
        if "region" in df_proc.columns:
                regions = ["northeast", "northwest", "southeast", "southwest"]
                df_proc["region"] = df_proc["region"].astype(str).str.lower()
                for r in regions:
                        df_proc[f"region_{r}"] = (df_proc["region"] == r).astype(int)
                df_proc = df_proc.drop(columns=["region"])

        # Ensure all expected columns exist (fill missing with zeros)
        expected = ["age", "sex", "bmi", "children", "smoker",
                                "region_northeast", "region_northwest", "region_southeast", "region_southwest"]
        for col in expected:
                if col not in df_proc.columns:
                        df_proc[col] = 0

        # Reorder to canonical ordering
        X = df_proc[expected].copy()

        if return_target:
                return X, y
        return X
