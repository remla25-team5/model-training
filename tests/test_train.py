import numpy as np
from sklearn.utils.validation import check_is_fitted
from sklearn.naive_bayes import GaussianNB
from model_training.modeling.train import create_pipeline_and_train, gaussiannb_classify
# bandit: disable=B101  (asserts are fine in this test)

# Test  ML infrastructure Infra 2: Model specification code is unit tested


def test_create_pipeline_and_train_basic():
    """Test the basic functionality of the create_pipeline_and_train function."""
    np.random.seed(42)
    X = np.random.rand(20, 5)
    y = np.random.randint(0, 2, size=20)
    classifier = GaussianNB()
    param_grid = {'classifier__var_smoothing': [1e-9]}
    score, estimator = create_pipeline_and_train(X, y, classifier, param_grid, cv_folds=2)
    assert 0.0 <= score <= 1.0
    check_is_fitted(estimator)


def test_gaussiannb_classify_basic():
    """Test the basic functionality of the gaussiannb_classify function."""
    np.random.seed(42)
    X = np.random.rand(20, 5)
    y = np.random.randint(0, 2, size=20)
    score, estimator = gaussiannb_classify(X, y, cv_folds=2)
    assert 0.0 <= score <= 1.0
    check_is_fitted(estimator)
