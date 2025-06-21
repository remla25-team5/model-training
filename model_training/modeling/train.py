import pickle
from pathlib import Path
import joblib

# from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
# from sklearn.naive_bayes import GaussianNB
# You can import and use other models similarly:
# from sklearn.linear_model import SGDClassifier, LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.svm import SVC

from loguru import logger


def create_pipeline_and_train(data, labels, classifier, param_grid, cv_folds, random_state=42):
    """
    Creates a pipeline and performs GridSearchCV to find the best model.

    Returns the best score and the best estimator found.
    """
    pipeline = Pipeline([('classifier', classifier)])

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    scoring = 'accuracy'

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=False,
        verbose=1,
    )

    logger.info(f"Starting GridSearchCV with {cv_folds} folds...")
    logger.info(f"Parameter Grid: {param_grid}")

    grid_search.fit(data, labels)

    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    best_estimator = grid_search.best_estimator_

    logger.success("GridSearchCV Complete")
    logger.info(f"Best Parameters: {best_params}")
    logger.info(f"Best Accuracy: {best_score:.6f}")

    return best_score, best_estimator


# def gaussiannb_classify(data, labels, cv_folds, random_state=42):
#     """
#     Trains a Gaussian Naive Bayes classifier using the specified parameter grid.

#     Returns the best score and the best trained GaussianNB model.
#     """
#     classifier = GaussianNB()
#     param_grid = {
#         'classifier__var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
#     }

#     best_score, best_estimator = create_pipeline_and_train(
#         data,
#         labels,
#         classifier,
#         param_grid,
#         cv_folds,
#         random_state=random_state
#     )

#     return best_score, best_estimator

def logisticregression_classify(data, labels, cv_folds, random_state=42):
    """
    Trains a LogisticRegression classifier using the specified parameter grid.

    Returns the best score and the best trained LogisticRegression model.
    """
    classifier = LogisticRegression(random_state=42, max_iter=1000)
    param_grid = {
        'classifier__C': [0.1, 1, 10, 100],
        'classifier__solver': ['liblinear', 'saga']
    }

    best_score, best_estimator = create_pipeline_and_train(
        data,
        labels,
        classifier,
        param_grid,
        cv_folds,
        random_state=random_state
    )

    return best_score, best_estimator


def main(
    data_path: Path = Path(__file__).parent.parent.parent / "data/processed/X_train.pkl",
    labels_path: Path = Path(__file__).parent.parent.parent / "data/processed/y_train.pkl",
    model_out_path: Path = Path(__file__).parent.parent.parent / "models/c2_Classifier_Sentiment_Model.joblib",
    metrics_path: Path = Path(__file__).parent.parent.parent / "metrics.json",
):
    """
    Loads training data, trains a Logistic Regression model, and saves the model and metrics.
    """
    logger.info("Loading data and labels...")
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    with open(labels_path, "rb") as f:
        labels = pickle.load(f)

    logger.info("Starting training with Logistic Regression...")
    best_score, best_estimator = logisticregression_classify(data, labels, cv_folds=5)

    model_out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_estimator, model_out_path)
    logger.success(f"Saved trained model to {model_out_path}")

    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(f'{{"accuracy": {best_score:.4f}}}')
    logger.success(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
