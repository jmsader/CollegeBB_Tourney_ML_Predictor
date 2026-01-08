from GLOBS import DATA_SLUG
import numpy as np
import csv
from xgboost import XGBClassifier, plot_importance
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import json


MODEL_TRANS_SLUG = "model_transparency/"


def get_data_split(split_name):
    """
    Load and prepare data split (train, validation, or test) from CSV file.

    Args:
        split_name: Name of the data split to load (e.g., 'training', 'val', 'test')

    Returns:
        X: Feature matrix as float array
        y: Target labels as float array
        feature_names: List of feature names from the CSV
    """
    with open(f"{DATA_SLUG}{split_name}_data.csv", "r") as f:
        # Read all rows from CSV into numpy array
        reader = np.array(list(csv.reader(f)))
        # Skip header row
        X = reader[1:]
        # Extract labels from second-to-last column
        y = X[:, -2]
        # Remove label column from feature matrix
        X = np.delete(X, -2, axis=1)
        # Extract feature names (starting from row 2)
        feature_names = reader[2:]
    return X.astype(float), y.astype(float), feature_names


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
        col_sample_bytree=hps["col_sample_bytree"],
        reg_lambda=hps["lambda"],
        reg_alpha=hps["alpha"],
    )


def test(model, X_test, y_test):
    """
    Evaluate the trained model on test data and generate feature importance plot.

    Args:
        model: Trained XGBoost model
        X_test: Test feature matrix
        y_test: Test labels
    """
    # Generate predictions on test set
    test_preds = model.predict(X_test)
    # Calculate and store accuracy
    accuracy = accuracy_score(y_test, test_preds)

    # Create and save feature importance visualization
    plot_importance(model)
    plt.savefig(f"{MODEL_TRANS_SLUG}feature_importance.png")


def tune(X_train, y_train, X_val, y_val):
    """
    Perform grid search hyperparameter tuning using validation set.

    Args:
        X_train: Training feature matrix
        y_train: Training labels
        X_val: Validation feature matrix
        y_val: Validation labels

    Returns:
        Dictionary containing best hyperparameter values
    """
    best_score = float("-inf")
    best_hParams = {}

    # Grid search over hyperparameter space
    for learning_rate in np.linspace(0.01, 0.3, 2):
        for max_depth in np.linspace(3, 10, 4, dtype=int):
            for n_estimators in np.linspace(100, 1000, 4, dtype=int):
                for subsample in np.linspace(0.5, 1, 2):
                    for colsample_bytree in np.linspace(0.3, 1, 2):
                        for Lambda in np.linspace(0, 10, 3):
                            for alpha in np.linspace(0, 10, 3):

                                # Train model with current hyperparameter combination
                                model = XGBClassifier(
                                    max_depth=max_depth,
                                    n_estimators=n_estimators,
                                    subsample=subsample,
                                    colsample_bytree=colsample_bytree,
                                    reg_lambda=Lambda,
                                    reg_alpha=alpha,
                                    learning_rate=learning_rate,
                                )
                                model.fit(X_train, y_train)

                                # Evaluate on validation set using ROC-AUC
                                val_pred_probs = model.predict_proba(X_val)
                                score = roc_auc_score(
                                    y_val, val_pred_probs, multi_class="ovr"
                                )
                                # Update best hyperparameters if current score is better
                                if score > best_score:
                                    best_score = score
                                    best_hParams["learning_rate"] = learning_rate
                                    best_hParams["max_depth"] = max_depth
                                    best_hParams["n_estimators"] = n_estimators
                                    best_hParams["subsample"] = subsample
                                    best_hParams["col_sample_bytree"] = colsample_bytree
                                    best_hParams["lambda"] = Lambda
                                    best_hParams["alpha"] = alpha

    return best_hParams


def main():
    """
    Main training pipeline: load data, tune hyperparameters, train final model, and evaluate.
    """
    # Load train, validation, and test datasets
    X_train, y_train, _ = get_data_split("training")
    X_val, y_val, _ = get_data_split("val")
    X_test, y_test, feature_names = get_data_split("test")

    # Perform hyperparameter tuning using grid search
    hyper_params = tune(X_train, y_train, X_val, y_val)
    # Save optimal hyperparameters to file
    with open(f"{MODEL_TRANS_SLUG}opt_hyper_params.txt", "w+") as f:
        json.dump(hyper_params, f)
    # Construct model with optimal hyperparameters
    model = construct_opt_model(hyper_params)
    # Set feature names for model interpretability
    model.get_booster().feature_names = feature_names
    # Train final model on training data
    model.fit(X_train, y_train)

    # Evaluate model on test set
    test(model, X_test, y_test)


if __name__ == "__main__":
    main()
