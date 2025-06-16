from pathlib import Path
import pickle
import json
import joblib

from loguru import logger
import typer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

app = typer.Typer()

# Default paths relative to the root of the repo
DEFAULT_TEST_DATA_PATH = Path(__file__).parent.parent / "data/processed/X_test.pkl"
DEFAULT_TEST_LABELS_PATH = Path(__file__).parent.parent / "data/processed/y_test.pkl"
DEFAULT_MODEL_PATH = Path(__file__).parent.parent / "models/c2_Classifier_Sentiment_Model.joblib"
DEFAULT_METRICS_OUTPUT_PATH = Path(__file__).parent.parent / "metrics_eval.json"


@app.command()
def main(
    test_data_path: Path = DEFAULT_TEST_DATA_PATH,
    test_labels_path: Path = DEFAULT_TEST_LABELS_PATH,
    model_path: Path = DEFAULT_MODEL_PATH,
    metrics_output_path: Path = DEFAULT_METRICS_OUTPUT_PATH,
):
    """
    Loads the trained sentiment classification model and evaluates it on the test set.
    Outputs accuracy, confusion matrix, and classification report.
    """
    logger.info(f"Loading test data from {test_data_path}")
    with open(test_data_path, "rb") as f:
        X_test = pickle.load(f)

    logger.info(f"Loading test labels from {test_labels_path}")
    with open(test_labels_path, "rb") as f:
        y_test = pickle.load(f)

    logger.info(f"Loading model from {model_path}")
    model = joblib.load(model_path)

    logger.info("Generating predictions...")
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred).tolist()
    class_report = classification_report(y_test, y_pred, output_dict=True)

    logger.success(f"Accuracy: {accuracy:.4f}")
    logger.info("Confusion Matrix:")
    logger.info(conf_matrix)
    logger.info("Classification Report:")
    logger.info(classification_report(y_test, y_pred))

    logger.info(f"Saving metrics to {metrics_output_path}")
    with open(metrics_output_path, "w", encoding="utf-8") as f:
        json.dump({
            "accuracy": accuracy,
            "confusion_matrix": conf_matrix,
            "classification_report": class_report,
        }, f, indent=4)

    logger.success("Evaluation completed and metrics saved.")


if __name__ == "__main__":
    app()
