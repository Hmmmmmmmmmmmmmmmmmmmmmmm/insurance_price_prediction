import numpy as np
import pandas as pd
import joblib
# from preprocessing import get_processed_data
from custom_model import Linear_Regression_Self
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from custom_model import StandardScalerSelf, Linear_Regression_Self
from sklearn.preprocessing import PolynomialFeatures

def pipeline_lr():
    pipeline_lr = Pipeline([
        ('scaler',StandardScaler()),
        ('mlr',LinearRegression())
    ])
    return pipeline_lr


def pipeline_self_lr():
    pipeline_self_lr = Pipeline([
        ('scaler',StandardScalerSelf()),
        ('mlr',Linear_Regression_Self())
    ])
    return pipeline_self_lr


def pipe_poly(degree=2):
    pipe_poly = Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('scaler', StandardScaler()),
        ('lr', LinearRegression())
    ])
    return pipe_poly

# def train_sklearn_lr():
#     X, y = get_processed_data()

#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42
#     )

#     model = LinearRegression()
#     model.fit(X_train, y_train)

#     joblib.dump(model, "../models/pipeline_lr.joblib")
#     joblib.dump(X_train, "../models/x_train.joblib")
#     joblib.dump(X_test, "../models/x_test.joblib")
#     joblib.dump(y_train, "../models/y_train.joblib")
#     joblib.dump(y_test, "../models/y_test.joblib")

#     return model

# # def train_custom_self():
# #     X, y = get_processed_data()

# #     X_train, X_test, y_train, y_test = train_test_split(
# #         X, y, test_size=0.2, random_state=42
# #     )

# #     self_model = Linear_Regression_Self()
# #     self_model.fit_normal(X_train, y_train)

# #     joblib.dump(self_model, "../models/pipe_poly.joblib")

# #     return self_model


# def train_custom_self():
#     X, y = get_processed_data()

#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42
#     )

#     self_model = Linear_Regression_Self()
#     self_model.fit_normal(X_train, y_train)

#     joblib.dump(self_model, "../models/pipe_poly.joblib")

#     return self_model


# def train_all():
#     print("Training Sklearn LR...")
#     train_sklearn_lr()

#     print("Training Self Model...")
#     train_custom_self()

#     print("All Training Completed")
