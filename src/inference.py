import numpy as np
import csv
from xgboost import XGBClassifier
import json

# File paths for input data and output predictions
TO_PREDICT_PATH = "to_predict.csv"
PREDICTIONS_PATH = "predictions.txt"


def get_data_splits(path):
    """
    Load and parse CSV data for model training or inference.

    Args:
        path: Path to the CSV file to load

    Returns:
        Tuple of (features, labels, team_names) as numpy arrays
    """
    with open(path, "r") as f:
        reader = np.array(list(csv.reader(f)))
        X = reader[1:]  # Skip header row
        y = X[:, -2]  # Extract labels from second-to-last column
        names = X[:, 0]  # Extract team names from first column

        # Remove team names column; for prediction data also remove labels column
        if path == TO_PREDICT_PATH:
            X = X[:, 1:]
        else:
            X = np.delete(X, -2, axis=1)

    return X.astype(float), y.astype(float), names


def construct_opt_model(hps):
    """
    Construct an XGBoost classifier with optimized hyperparameters.

    Args:
        hps: Dictionary containing hyperparameter values

    Returns:
        Configured XGBClassifier instance
    """
    return XGBClassifier(
        max_depth=hps["max_depth"],
        n_estimators=hps["n_estimators"],
        subsample=hps["subsample"],
        colsample_bytree=hps["col_sample_bytree"],
        reg_lambda=hps["lambda"],
        reg_alpha=hps["alpha"],
    )


def inference(model, X_inf, names):
    """
    Generate predictions and write results to file.

    Args:
        model: Trained XGBoost model
        X_inf: Feature matrix for inference
        names: Array of team names corresponding to predictions
    """
    # Generate predictions for all teams
    preds = model.predict(X_inf)
    preds_str_list = names.tolist()

    # Mapping of prediction values to tournament round descriptions
    pred_map = {
        "0": "Not in NCAAT",
        "1": "Round of 68",
        "2": "Round of 64",
        "3": "Round of 32",
        "4": "Sweet 16",
        "5": "Elite 8",
        "6": "Final 4",
        "7": "Runner Up",
        "8": "Champions",
    }

    # Format each prediction as "Team Name: Round"
    for r in range(0, len(names)):
        preds_str_list[r] += ": " + pred_map[str(preds[r])] + "\n"

    # Write predictions to output file
    with open(PREDICTIONS_PATH, "w+") as f:
        f.writelines(preds_str_list)


def main():
    """
    Main execution function: load data, train model, and generate predictions.
    """
    # Load training data
    X_train, y_train, _ = get_data_splits("model_training/data/cleaned_cbb.csv")

    # Load data for inference
    X_inf, _, names = get_data_splits(TO_PREDICT_PATH)

    # Load optimized hyperparameters from JSON file
    with open("model_transparency/opt_hyper_params.json", "r") as f:
        hyper_params = json.load(f)

    # Construct and train the model
    model = construct_opt_model(hyper_params)
    model.fit(X_train, y_train)

    # Generate and save predictions
    inference(model, X_inf, names)


if __name__ == "__main__":
    main()
