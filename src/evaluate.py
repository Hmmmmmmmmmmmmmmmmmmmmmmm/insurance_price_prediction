import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def eval_mse_mae_r2(y_test,y_pred):
    mse = mean_squared_error(y_test, y_pred)
    print("MSE:", mse)

    mae = mean_absolute_error(y_test,y_pred)
    print ("MAE:", mae)

    r2scorer = r2_score(y_test,y_pred)
    print(r2scorer)

    return mse, mae, r2scorer
def residual_analysis(pipeline, X, y):
    """
    Plot residuals and highlight patterns.

    Parameters:
    - pipeline: trained regression pipeline
    - X, y: data to analyze (can be train or test)
    """
    import matplotlib.pyplot as plt
    import numpy as np

    y_pred = pipeline.predict(X)
    residuals = y - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(12,5))

    # Residuals vs Predicted
    axes[0].scatter(y_pred, residuals, alpha=0.5, color='purple')
    axes[0].axhline(0, color='red', linestyle='--')
    axes[0].set_xlabel("Predicted Values")
    axes[0].set_ylabel("Residuals")
    axes[0].set_title("Residuals vs Predicted")

    # Histogram of residuals
    axes[1].hist(residuals, bins=20, color='orange', edgecolor='k', alpha=0.7)
    axes[1].set_xlabel("Residuals")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Residual Distribution")

    plt.tight_layout()
    plt.show()

    print(f"Residuals mean: {np.mean(residuals):.4f} | Residuals std: {np.std(residuals):.4f}")

def assumption_checks(pipeline, X, y):
    """
    Explicitly check common linear regression assumptions.

    Parameters:
    - pipeline: trained regression pipeline
    - X, y: data to check
    """
    import matplotlib.pyplot as plt
    import scipy.stats as stats
    import numpy as np

    y_pred = pipeline.predict(X)
    residuals = y - y_pred

    # 1. Linearity: Residuals vs Predicted
    plt.figure(figsize=(6,4))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel("Predicted")
    plt.ylabel("Residuals")
    plt.title("Linearity Check")
    plt.show()

    # 2. Normality of residuals: Q-Q plot
    plt.figure(figsize=(6,4))
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title("Normality of Residuals (Q-Q plot)")
    plt.show()

    # 3. Homoscedasticity: residual variance check
    plt.figure(figsize=(6,4))
    plt.scatter(y_pred, np.abs(residuals), alpha=0.5, color='orange')
    plt.xlabel("Predicted")
    plt.ylabel("|Residuals|")
    plt.title("Homoscedasticity Check")
    plt.show()

    # 4. Independence check (optional for time-series)
    print("Assumptions checked: Linearity, Normality, Homoscedasticity.")

def multicollinearity_check(X):
    """
    Check multicollinearity using VIF (Variance Inflation Factor).

    Parameters:
    - X: feature matrix (preferably DataFrame for feature names)
    """
    import pandas as pd
    import numpy as np
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=[f"Feature {i}" for i in range(X.shape[1])])

    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    print("\n=== Multicollinearity Check (VIF) ===")
    print(vif_data.sort_values(by="VIF", ascending=False))

    return vif_data

def error_segmentation(pipeline, X, y, feature_idx=0):
    """
    Segment errors along a selected feature.

    Parameters:
    - pipeline: trained pipeline
    - X, y: dataset
    - feature_idx: index of feature to segment errors
    """
    import matplotlib.pyplot as plt
    import numpy as np

    y_pred = pipeline.predict(X)
    residuals = y - y_pred

    feature_values = X[:, feature_idx] if isinstance(X, np.ndarray) else X.iloc[:, feature_idx].values

    plt.figure(figsize=(6,4))
    plt.scatter(feature_values, residuals, alpha=0.5, color='green')
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel(f"Feature {feature_idx}")
    plt.ylabel("Residuals")
    plt.title(f"Error Segmentation by Feature {feature_idx}")
    plt.show()

    print(f"Mean residual for feature {feature_idx}: {np.mean(residuals):.4f}")

def analyze_model(pipeline, X_train, y_train, X_test, y_test, partial_dependence=False):
    """
    Visualize and interpret a trained regression pipeline.

    Works for sklearn-like pipelines or custom pipelines with Linear_Regression_Self.

    Parameters:
    - pipeline: sklearn-like pipeline with a final regressor
    - X_train, y_train: training data
    - X_test, y_test: test data
    - partial_dependence: bool, if True, plot simple 1-feature partial dependence plots
    """

    # ----------------------
    # 1️⃣ Predictions
    # ----------------------
    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)

    # ----------------------
    # 2️⃣ Get final model
    # ----------------------
    model = pipeline.steps[-1][1]

    # ----------------------
    # 3️⃣ Learning Curve (if available)
    # ----------------------
    if hasattr(model, 'losses'):
        plt.figure(figsize=(7,4))
        plt.plot(model.losses, color='navy')
        plt.title("Learning Curve (MSE)")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.grid(True, alpha=0.3)
        plt.show()

    # ----------------------
    # 4️⃣ Actual vs Predicted
    # ----------------------
    plt.figure(figsize=(6,6))
    plt.scatter(y_test, y_test_pred, alpha=0.5, color='teal')
    min_val, max_val = min(y_test.min(), y_test_pred.min()), max(y_test.max(), y_test_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Actual vs Predicted (Test Set)")
    plt.show()

    # ----------------------
    # 5️⃣ Residuals
    # ----------------------
    residuals = y_test - y_test_pred
    plt.figure(figsize=(6,4))
    plt.scatter(y_test_pred, residuals, alpha=0.5, color='purple')
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel("Predicted")
    plt.ylabel("Residuals")
    plt.title("Residuals vs Predicted")
    plt.show()

    # ----------------------
    # 6️⃣ Coefficients
    # ----------------------
    if hasattr(model, 'w_'):       # custom Linear_Regression_Self
        coefs = model.w_
        intercept = model.b_ if hasattr(model, 'b_') else 0
    elif hasattr(model, 'coef_'):  # sklearn LinearRegression
        coefs = model.coef_
        intercept = model.intercept_ if hasattr(model, 'intercept_') else 0
    else:
        coefs = None
        intercept = None

    if coefs is not None:
        # Ensure features match number of coefficients
        n_coefs = len(coefs)
        if isinstance(X_train, pd.DataFrame):
            if n_coefs == X_train.shape[1]:
                features = X_train.columns
            else:
                # Expanded features (e.g., PolynomialFeatures)
                features = [f"Feature {i}" for i in range(n_coefs)]
        else:
            features = [f"Feature {i}" for i in range(n_coefs)]

        coef_df = pd.DataFrame({"Feature": features, "Coefficient": coefs})
        coef_df = coef_df.sort_values(by="Coefficient", key=abs, ascending=False)
        print("\n=== Feature Coefficients ===")
        print(coef_df)
        print(f"Intercept: {intercept}")

        # Visualize
        plt.figure(figsize=(8,4))
        plt.bar(coef_df["Feature"], coef_df["Coefficient"], color='skyblue')
        plt.xticks(rotation=45)
        plt.ylabel("Coefficient Value")
        plt.title("Feature Coefficients")
        plt.show()


    # ----------------------
    # 7️⃣ Partial Dependence Plots (Optional)
    # ----------------------
    if partial_dependence:
        if isinstance(X_test, pd.DataFrame):
            features = X_test.columns
            X_arr = X_test.values
        else:
            features = [f"Feature {i}" for i in range(X_test.shape[1])]
            X_arr = X_test

        for idx, feat in enumerate(features):
            X_fixed = X_arr.copy()
            other_idxs = [i for i in range(X_arr.shape[1]) if i != idx]
            X_fixed[:, other_idxs] = X_fixed[:, other_idxs].mean(axis=0)
            y_pred_feat = pipeline.predict(X_fixed)

            plt.figure(figsize=(6,4))
            plt.plot(X_fixed[:, idx], y_pred_feat, color='orange')
            plt.xlabel(feat)
            plt.ylabel("Predicted Target")
            plt.title(f"Partial Dependence: {feat}")
            plt.grid(True, alpha=0.3)
            plt.show()

    # ----------------------
    # 8️⃣ Metrics
    # ----------------------
    print(f"Train MSE: {mean_squared_error(y_train, y_train_pred):.4f} | Test MSE: {mean_squared_error(y_test, y_test_pred):.4f}")
    print(f"Train R²: {r2_score(y_train, y_train_pred):.4f} | Test R²: {r2_score(y_test, y_test_pred):.4f}")

def dashboard_model(pipeline, X_train, y_train, X_test, y_test, model_name="Model", partial_dependence=False, normalize_coefs=True, return_figure: bool = False):
    """
    Dashboard-style visualization for a regression pipeline.
    Includes learning curve, actual vs predicted, residuals, coefficients, metrics, and optional partial dependence.

    Parameters:
    - pipeline: sklearn-like pipeline with a final regressor
    - X_train, y_train, X_test, y_test: data
    - model_name: string to title the dashboard
    - partial_dependence: bool, whether to show simple 1-feature partial dependence plots
    - normalize_coefs: bool, scale coefficients for easier visualization
    """
    # ----------------------
    # 1️⃣ Predictions
    # ----------------------
    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)

    # ----------------------
    # 2️⃣ Metrics
    # ----------------------
    mse_train = mean_squared_error(y_train, y_train_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)
    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)

    # ----------------------
    # 3️⃣ Extract model & coefficients
    # ----------------------
    model = None
    coefs = None
    if 'mlr_self' in pipeline.named_steps:
        model = pipeline.named_steps['mlr_self']
    elif 'regressor' in pipeline.named_steps:
        model = pipeline.named_steps['regressor']

    if model and hasattr(model, 'w_'):
        coefs = model.w_
        if normalize_coefs:
            coefs = coefs / np.max(np.abs(coefs))
        if isinstance(X_train, pd.DataFrame):
            features = X_train.columns
        else:
            features = [f"Feature {i}" for i in range(X_train.shape[1])]
    else:
        features = []

    # ----------------------
    # 4️⃣ Dashboard layout (2x3 grid)
    # ----------------------
    nrows, ncols = 2, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(20,12))
    fig.suptitle(f"{model_name} Dashboard\nTrain R²: {r2_train:.3f}, Test R²: {r2_test:.3f} | Train MSE: {mse_train:.2f}, Test MSE: {mse_test:.2f}", fontsize=18)

    # 4a️⃣ Learning Curve
    ax = axes[0,0]
    if model and hasattr(model, 'losses'):
        ax.plot(model.losses, color='navy')
        ax.set_title("Learning Curve (MSE)")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5,0.5,"No Learning Curve",ha='center',va='center',fontsize=14)
        ax.axis('off')

    # 4b️⃣ Actual vs Predicted
    ax = axes[0,1]
    ax.scatter(y_test, y_test_pred, alpha=0.5, color='teal')
    min_val, max_val = min(y_test.min(), y_test_pred.min()), max(y_test.max(), y_test_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--')
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Actual vs Predicted (Test)")

    # 4c️⃣ Residuals
    ax = axes[0,2]
    residuals = y_test - y_test_pred
    ax.scatter(y_test_pred, residuals, alpha=0.5, color='purple')
    ax.axhline(0, color='red', linestyle='--')
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residuals")
    ax.set_title("Residuals vs Predicted")

    # 4d️⃣ Feature Coefficients
    ax = axes[1,0]
    if coefs is not None and len(coefs) > 0:
        ax.bar(features, coefs, color='skyblue')
        ax.set_xticklabels(features, rotation=45, ha='right')
        ax.set_ylabel("Coefficient Value (Normalized)" if normalize_coefs else "Coefficient Value")
        ax.set_title("Feature Coefficients")
    else:
        ax.text(0.5,0.5,"No Coefficients",ha='center',va='center',fontsize=14)
        ax.axis('off')

    # 4e️⃣ Optional: Partial Dependence Plots (single feature per subplot)
    ax = axes[1,1]
    if partial_dependence and len(features) > 0:
        feat_idx = 0  # first feature
        X_arr = X_test.values if isinstance(X_test, pd.DataFrame) else X_test.copy()
        X_fixed = X_arr.copy()
        other_idxs = [i for i in range(X_arr.shape[1]) if i != feat_idx]
        X_fixed[:, other_idxs] = X_fixed[:, other_idxs].mean(axis=0)
        y_pred_feat = pipeline.predict(X_fixed)
        ax.plot(X_fixed[:, feat_idx], y_pred_feat, color='orange')
        ax.set_xlabel(features[feat_idx])
        ax.set_ylabel("Predicted Target")
        ax.set_title(f"Partial Dependence: {features[feat_idx]}")
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5,0.5,"Partial Dependence\n(optional)",ha='center',va='center',fontsize=14)
        ax.axis('off')

    # 4f️⃣ Empty / placeholder for layout
    axes[1,2].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if return_figure:
        return fig
    plt.show()
