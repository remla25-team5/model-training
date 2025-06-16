import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_is_fitted
from model_training.modeling.train import create_pipeline_and_train, logisticregression_classify

# Test  ML infrastructure Infra 2: Model specification code is unit tested


def test_create_pipeline_and_train_basic():
    """Test the basic functionality of the create_pipeline_and_train function."""
    np.random.seed(42)
    X = np.random.rand(20, 5)
    y = np.random.randint(0, 2, size=20)
    classifier = LogisticRegression()
    param_grid = {'classifier__C': [0.1, 1.0, 10.0]}
    score, estimator = create_pipeline_and_train(X, y, classifier, param_grid, cv_folds=2)
    assert 0.0 <= score <= 1.0
    check_is_fitted(estimator)


def test_gaussiannb_classify_basic():
    """Test the basic functionality of the gaussiannb_classify function."""
    np.random.seed(42)
    X = np.random.rand(20, 5)
    y = np.random.randint(0, 2, size=20)
    score, estimator = logisticregression_classify(X, y, cv_folds=2)
    assert 0.0 <= score <= 1.0
    check_is_fitted(estimator)


def test_non_determinism_robustness():
    """Test that the model is robust against non-deterministic behavior."""
    np.random.seed(42)
    X = np.random.rand(1000, 5)
    y = np.random.randint(0, 2, size=1000)
    original_score, _ = logisticregression_classify(X, y, cv_folds=5)

    for seed in [1, 2]:
        np.random.seed(seed)
        X_new = np.random.rand(1000, 5)
        y_new = np.random.randint(0, 2, size=1000)
        new_score, _ = logisticregression_classify(X_new, y_new, cv_folds=5)
        assert abs(original_score - new_score) <= 0.03
