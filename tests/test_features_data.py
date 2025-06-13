import numpy as np
import pandas as pd
import pickle
import pytest
import requests
import time

from pathlib import Path

# bandit: disable=B101  (asserts are fine in this test)

# Test Features and Data:
# 1. Test that the distributions of each feature match our expectations.
# Data 3: No feature’s cost is too much


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


def test_raw_dataset_distribution(raw_dataset):
    """
    Test the distribution characteristics of the raw dataset.
    """
    # Check dataset size. It should not be too small (less than 100 rows) or too large (more than 10,000 rows).
    assert len(raw_dataset) > 100, f"Dataset seems too small: {len(raw_dataset)} rows"
    assert len(raw_dataset) < 10000, f"Dataset seems too large: {len(raw_dataset)} rows"

    # Check review length distribution. The mean length should be reasonable (between 10 and 1000 characters).
    review_lengths = raw_dataset['Review'].str.len()
    mean_length = review_lengths.mean()
    assert 10 < mean_length < 1000, f"Average review length seems unusual: {mean_length}"

    # Check for very short or very long reviews. The ratio of very short (< 10 characters)
    # and very long (> 500 characters) reviews should be low.
    very_short = np.sum(review_lengths < 10)
    very_long = np.sum(review_lengths > 500)

    short_ratio = very_short / len(raw_dataset)
    long_ratio = very_long / len(raw_dataset)

    assert short_ratio < 0.1, f"Too many very short reviews: {short_ratio:.2%}"
    assert long_ratio < 0.1, f"Too many very long reviews: {long_ratio:.2%}"

    # Check label distribution. The ratio of positive labels (1) should be between 40% and 60%.
    label_counts = raw_dataset['Liked'].value_counts()
    positive_ratio = label_counts.get(1, 0) / len(raw_dataset)

    assert 0.4 < positive_ratio < 0.6, f"Label distribution seems imbalanced: {positive_ratio:.2%} positive"


def test_feature_cost_analysis(load_vectorizer):
    """
    Data 3: Test that no feature's cost is too high.
    Measures extraction time cost and cost of memory usage of features.
    """
    # Measure feature extraction time cost
    start_time = time.time()
    test_corpus = ["This is a good restaurant."]
    features = load_vectorizer.transform(test_corpus).toarray()
    feature_extraction_time = time.time() - start_time

    # Feature extraction should be fast (< 1 second for single review)
    assert feature_extraction_time < 1.0, f"Feature extraction too slow: {feature_extraction_time:.3f}s"

    # Measure memory usage of features
    feature_memory_mb = features.nbytes / (1024 * 1024)
    feature_memory_per_sample = feature_memory_mb / features.shape[0]

    # Memory usage should be reasonable (< 15MB per 1000 samples)
    memory_per_1k_samples = feature_memory_per_sample * 1000
    assert memory_per_1k_samples < 15, f"Features use too much memory: {memory_per_1k_samples:.2f}MB per 1000 samples"
