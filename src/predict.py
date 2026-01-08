# import sys
# import joblib
# import pandas as pd

# MODEL_PATH = "models/pipeline_self_mlr.joblib"

# def load_input(path):
#     df = pd.read_csv(path)
#     return df

# def main(csv_path):
#     model = joblib.load(MODEL_PATH)
#     df = load_input(csv_path)
#     preds = model.predict(df)
#     out = pd.DataFrame(preds, columns=["charges_pred"])
#     print(out.to_csv(index=False))

# if __name__ == "__main__":
#     if len(sys.argv) < 2:
#         print("Usage: python predict.py input_row.csv")
#     else:
#         main(sys.argv[1])

# src/predict.py

import joblib
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "pipeline_self_mlr.joblib"

# - pipeline_lr.joblib
# - pipe_poly.joblib

def predict(input_data: dict, model_path: Path = MODEL_PATH) -> float:
    """
    Predict insurance price for a single input.

    Parameters
    ----------
    input_data : dict
        Example:
        {
            "age": 35,
            "sex": "male",
            "bmi": 27.5,
            "children": 2,
            "smoker": "no",
            "region": "southwest"
        }

    model_path : Path
        Path to trained pipeline (.joblib)

    Returns
    -------
    float
        Predicted insurance cost
    """
    model = joblib.load(model_path)
    X = pd.DataFrame([input_data])

    # Predict
    prediction = model.predict(X)

    return float(prediction[0])

def ask_user(prompt, cast_func=str, choices=None):
    while True:
        value = input(prompt)
        try:
            value = cast_func(value)
            if choices and value not in choices:
                raise ValueError
            return value
        except ValueError:
            print("❌ Invalid input, try again.")

if __name__ == "__main__":
    print("\n=== Insurance Price Prediction ===\n")

    user_input = {
        "age": ask_user("Age: ", int),
        "sex": ask_user("Sex (male/female): ", str, ["male", "female"]),
        "bmi": ask_user("BMI: ", float),
        "children": ask_user("Number of children: ", int),
        "smoker": ask_user("Smoker (yes/no): ", str, ["yes", "no"]),
        "region": ask_user(
            "Region (northeast/northwest/southeast/southwest): ",
            str,
            ["northeast", "northwest", "southeast", "southwest"]
        ),
    }
    # user_input['sex']=("yes")?0:1
    user_input['smoker'] = 0 if user_input['smoker'] == "yes" else 1
    user_input['sex'] = 0 if user_input['sex'] == "female" else 1
    # Remove the original 'region' entry
    region = user_input.pop('region')

    # Initialize all region columns to 0
    user_input['region_northeast'] = 0
    user_input['region_northwest'] = 0
    user_input['region_southeast'] = 0
    user_input['region_southwest'] = 0

    # Set the correct region to 1
    if region == "northeast":
        user_input['region_northeast'] = 1
    elif region == "northwest":
        user_input['region_northwest'] = 1
    elif region == "southeast":
        user_input['region_southeast'] = 1
    elif region == "southwest":
        user_input['region_southwest'] = 1
    pred = predict(user_input)

    print("\n===================================")
    print(f"💰 Predicted insurance cost: ${pred:,.2f}")
    print("===================================\n")



# if __name__ == "__main__":

#     sample_input = {
#         "age": 35,
#         "sex": "male",
#         "bmi": 27.5,
#         "children": 2,
#         "smoker": "no",
#         "region": "southwest"
#     }

#     pred = predict(sample_input)

#     print("===================================")
#     print(" Insurance Price Prediction")
#     print("===================================")
#     print(f"Predicted cost: ${pred:,.2f}")
