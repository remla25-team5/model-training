import json
import numpy as np
import pandas as pd
import pickle
import pytest
import requests
import subprocess
import sys
import time

from lib_ml.preprocessing import preprocess_dataset
from model_training.modeling.train import gaussiannb_classify
from pathlib import Path
# bandit: disable=B101  (asserts are fine in this test)

# Test ML Infrastructure:
# Infra 1: Training is reproducible
# Infra 3: The full ML pipeline is integration tested.


@pytest.fixture
def raw_dataset():
    """
    Fixture that provides the raw restaurant reviews dataset from the URL.
    """
    base_url = "https://storage.googleapis.com/remla-group-5-unique-bucket"
    filename = "a1_RestaurantReviews_HistoricDump.tsv"

    # Check if data already exists locally
    data_dir = Path(__file__).parent.parent / "data" / "raw"
    file_path = data_dir / filename

    if not file_path.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        url = f"{base_url}/{filename}"

        # Retry logic
        max_retries = 3
        delay = 5

        for attempt in range(max_retries):
            try:
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                with open(file_path, "wb") as f:
                    f.write(response.content)
                break
            except requests.exceptions.RequestException:
                if attempt < max_retries - 1:
                    time.sleep(delay * (2 ** attempt))
                else:
                    raise

    # Load and return the dataset
    dataset = pd.read_csv(file_path, delimiter='\t', quoting=3)
    return dataset


@pytest.fixture
def load_vectorizer():
    """
    Fixture that loads the pre-trained CountVectorizer for testing.
    """
    vectorizer_path = Path(__file__).parent.parent / "models" / "c1_BoW_Sentiment_Model.pkl"
    if not vectorizer_path.exists():
        raise FileNotFoundError(f"Vectorizer file not found: {vectorizer_path}")

    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    return vectorizer


# Infra 1: Training is reproducible
def test_training_reproducibility_same_seed(raw_dataset, load_vectorizer):
    """
    Test that training with the same random seed produces identical results.
    """

    corpus, labels = preprocess_dataset(raw_dataset)
    features = load_vectorizer.fit_transform(corpus).toarray()

    # Train model1
    best_score1, best_estimator1 = gaussiannb_classify(features, labels, cv_folds=5, random_state=42)

    # Train model2
    best_score2, best_estimator2 = gaussiannb_classify(features, labels, cv_folds=5, random_state=42)

    # Results should be identical
    assert best_score1 == best_score2, f"Scores differ: {best_score1} vs {best_score2}"

    # Model parameters should be the same
    params1 = best_estimator1.get_params()
    params2 = best_estimator2.get_params()

    # Check key parameters are identical
    assert params1['classifier__var_smoothing'] == params2['classifier__var_smoothing']


# Infra 3: The full ML pipeline is integration tested
def test_full_pipeline_integration():
    """
    Integration test that runs the complete ML pipeline.
    Tests: dataset.py -> transform.py -> train.py -> evaluation.py
    """
    # Get the project root directory
    project_root = Path(__file__).parent.parent

    # Step 1: Run dataset.py to download data
    print("Step 1: Running dataset.py...")
    result = subprocess.run([
        sys.executable, "-m", "model_training.dataset"
    ], cwd=project_root, capture_output=True, text=True)

    assert result.returncode == 0, f"dataset.py failed: {result.stderr}"

    # Verify raw data exists
    raw_data_path = project_root / "data" / "raw" / "a1_RestaurantReviews_HistoricDump.tsv"
    assert raw_data_path.exists(), "Raw data file not created by dataset.py"

    # Step 2: Run transform.py to process data
    print("Step 2: Running transform.py...")
    result = subprocess.run([
        sys.executable, "-m", "model_training.transform"
    ], cwd=project_root, capture_output=True, text=True)

    assert result.returncode == 0, f"transform.py failed: {result.stderr}"

    # Verify processed data exists
    processed_dir = project_root / "data" / "processed"
    X_train_path = processed_dir / "X_train.pkl"
    y_train_path = processed_dir / "y_train.pkl"
    X_test_path = processed_dir / "X_test.pkl"
    y_test_path = processed_dir / "y_test.pkl"

    assert X_train_path.exists(), "X_train.pkl not created by transform.py"
    assert y_train_path.exists(), "y_train.pkl not created by transform.py"
    assert X_test_path.exists(), "X_test.pkl not created by transform.py"
    assert y_test_path.exists(), "y_test.pkl not created by transform.py"

    # Verify vectorizer exists
    vectorizer_path = project_root / "models" / "c1_BoW_Sentiment_Model.pkl"
    assert vectorizer_path.exists(), "Vectorizer not created by transform.py"

    # Step 3: Run train.py to train model
    print("Step 3: Running train.py...")
    result = subprocess.run([
        sys.executable, "-m", "model_training.modeling.train"
    ], cwd=project_root, capture_output=True, text=True)

    assert result.returncode == 0, f"train.py failed: {result.stderr}"

    # Verify model was created (check if any model files exist)
    models_dir = project_root / "models"
    model_files = list(models_dir.glob("*.pkl"))
    assert len(model_files) > 0, "No model files created by train.py"

    # Step 4: Run evaluation.py
    print("Step 4: Running evaluation.py...")
    result = subprocess.run([
        sys.executable, "-m", "model_training.evaluation"
    ], cwd=project_root, capture_output=True, text=True)

    assert result.returncode == 0, f"evaluation.py failed: {result.stderr}"

    # Step 5: Verify the pipeline produced valid outputs
    print("Step 5: Verifying pipeline outputs...")

    # Load and verify processed data
    with open(X_train_path, 'rb') as f:
        X_train = pickle.load(f)
    with open(y_train_path, 'rb') as f:
        y_train = pickle.load(f)
    with open(X_test_path, 'rb') as f:
        X_test = pickle.load(f)
    with open(y_test_path, 'rb') as f:
        y_test = pickle.load(f)

    # Verify data shapes are as expected
    assert X_train.shape[0] == len(y_train), "Training data shape mismatch"
    assert X_test.shape[0] == len(y_test), "Test data shape mismatch"
    assert X_train.shape[1] == X_test.shape[1], "Feature dimension mismatch"
    assert X_train.shape[0] > 0, "No training samples"
    assert X_test.shape[0] > 0, "No test samples"

    # Verify labels are binary
    assert set(np.unique(y_train)).issubset({0, 1}), "Training labels not binary"
    assert set(np.unique(y_test)).issubset({0, 1}), "Test labels not binary"

    # Load and verify vectorizer
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    assert hasattr(vectorizer, 'vocabulary_'), "Vectorizer not properly fitted"
    assert len(vectorizer.vocabulary_) > 0, "Empty vocabulary"

    # Verify output metrics file exists
    metrics_output_path = project_root / "metrics_eval.json"
    assert metrics_output_path.exists(), "Metrics output file not created by evaluation.py"
    with open(metrics_output_path, 'r', encoding='utf-8') as f:
        metrics = json.load(f)
    assert 'accuracy' in metrics, "Accuracy not found in metrics"
    assert isinstance(metrics['accuracy'], float), "Accuracy should be a float"
    assert 0 <= metrics['accuracy'] <= 1, "Accuracy out of bounds (0-1)"
