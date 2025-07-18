import joblib
import numpy as np
import pandas as pd
import pickle
import psutil
import pytest
import requests
import time

from lib_ml.preprocessing import preprocess_dataset
from model_training.modeling.train import logisticregression_classify
from pathlib import Path

# Test Monitoring:
# Monitor 6: The model has not experienced a dramatic or slow-leak regressions in training speed,
#            serving latency, throughput, or RAM usage


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
def load_model():
    """
    Fixture that loads the pre-trained model for testing.
    """
    model_path = Path(__file__).parent.parent / "models" / "c2_Classifier_Sentiment_Model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    with open(model_path, 'rb') as f:
        model = joblib.load(f)
    return model


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


@pytest.fixture
def performance_baseline():
    """Fixture providing performance baselines for comparison."""
    return {
        'training_time_per_sample': 0.01,
        'inference_time_per_sample': 0.001,
        'memory_per_sample': 1.0,
        'throughput_min': 100,
        'model_size_max': 10.0
    }


# Monitor 6: Model computational performance monitoring
def test_training_speed_performance(raw_dataset, performance_baseline, load_vectorizer):
    """
    Measures training time and ensures it hasn't regressed significantly.
    """
    corpus, labels = preprocess_dataset(raw_dataset)
    features = load_vectorizer.fit_transform(corpus).toarray()

    # Measure training time
    start_time = time.time()

    # Train model
    logisticregression_classify(features, labels, cv_folds=5, random_state=42)

    training_time = time.time() - start_time
    training_time_per_sample = training_time / len(raw_dataset)

    # Check for dramatic regression (20x slower than baseline)
    dramatic_regression_threshold = performance_baseline['training_time_per_sample'] * 20
    assert training_time_per_sample < dramatic_regression_threshold, \
        f"Dramatic training speed regression detected: {training_time_per_sample:.4f}s per sample > \
            {dramatic_regression_threshold:.4f}s"

    # Check for slow-leak regression (10x slower than baseline)
    slow_leak_threshold = performance_baseline['training_time_per_sample'] * 10
    assert training_time_per_sample < slow_leak_threshold, \
        f"Slow-leak training speed regression detected: {training_time_per_sample:.4f}s per sample >\
              {slow_leak_threshold:.4f}s"

    # Log performance for monitoring
    print(f"Training performance: {training_time_per_sample:.4f}s per sample \
          (baseline: {performance_baseline['training_time_per_sample']:.4f}s)")


def test_serving_latency_performance(performance_baseline, load_vectorizer, load_model):
    """
    Measures prediction time for individual samples.
    """
    # Measure single-sample inference latency
    single_sample_text = ["This restaurant is great!"]
    single_sample = load_vectorizer.transform(single_sample_text).toarray()

    # Warm up the model (first prediction is often slower)
    load_model.predict(single_sample)

    # Measure actual latency
    latency_times = []
    for _ in range(100):
        start_time = time.time()
        load_model.predict(single_sample)
        latency_times.append(time.time() - start_time)

    avg_latency = np.mean(latency_times)

    # Check for dramatic latency regression (20x slower)
    dramatic_latency_threshold = performance_baseline['inference_time_per_sample'] * 20
    assert avg_latency < dramatic_latency_threshold, \
        f"Dramatic serving latency regression: {avg_latency:.6f}s >\
              {dramatic_latency_threshold:.6f}s"

    # Check for slow-leak latency regression (5x slower)
    slow_leak_latency_threshold = performance_baseline['inference_time_per_sample'] * 5
    assert avg_latency < slow_leak_latency_threshold, \
        f"Slow-leak serving latency regression: {avg_latency:.6f}s >\
              {slow_leak_latency_threshold:.6f}s"

    print(f"Serving latency: {avg_latency:.6f}s per sample\
           (baseline: {performance_baseline['inference_time_per_sample']:.6f}s)")


def test_throughput_performance(raw_dataset, performance_baseline, load_vectorizer, load_model):
    """
    Measures how many samples can be processed per second.
    """
    corpus, _ = preprocess_dataset(raw_dataset)
    features = load_vectorizer.fit_transform(corpus).toarray()

    # Measure batch throughput
    batch_size = 100
    test_features = features[:batch_size]

    assert len(test_features) == batch_size, "Test features size does not match batch size"

    # Run multiple predictions to get a more reliable timing
    num_runs = 5
    total_time = 0

    # Warm up the model first
    load_model.predict(test_features)

    for _ in range(num_runs):
        start_time = time.time()
        load_model.predict(test_features)
        total_time += time.time() - start_time

    # Use average time to calculate throughput, with a safety check
    batch_time = total_time / num_runs

    # Ensure batch_time is not zero to prevent division by zero
    if batch_time == 0:
        print("Warning: Batch time is zero, setting to a small value to avoid division by zero.")
        batch_time = 1e-6

    throughput = batch_size / batch_time

    # Check for dramatic throughput regression (10x slower)
    dramatic_throughput_threshold = performance_baseline['throughput_min'] / 10
    assert throughput > dramatic_throughput_threshold, \
        f"Dramatic throughput regression: {throughput:.2f} samples/s < {dramatic_throughput_threshold:.2f} samples/s"

    # Check for slow-leak throughput regression (2x slower)
    slow_leak_throughput_threshold = performance_baseline['throughput_min'] / 2
    assert throughput > slow_leak_throughput_threshold, \
        f"Slow-leak throughput regression: {throughput:.2f} samples/s < {slow_leak_throughput_threshold:.2f} samples/s"

    print(f"Throughput: {throughput:.2f} samples/s \
          (baseline minimum: {performance_baseline['throughput_min']} samples/s)")


def test_memory_usage_performance(raw_dataset, performance_baseline, load_vectorizer):
    """
    Monitors memory consumption during training and inference.
    """
    process = psutil.Process()

    # Prepare data
    corpus, labels = preprocess_dataset(raw_dataset)
    features = load_vectorizer.fit_transform(corpus).toarray()

    # Measure memory during training
    training_stat_memory = process.memory_info().rss / 1024 / 1024

    logisticregression_classify(features, labels, cv_folds=5, random_state=42)

    training_end_memory = process.memory_info().rss / 1024 / 1024
    training_memory_usage = training_end_memory - training_stat_memory

    # Measure memory per sample
    memory_per_sample = training_memory_usage / len(raw_dataset)

    # Check for dramatic memory regression (10x more memory)
    dramatic_memory_threshold = performance_baseline['memory_per_sample'] * 10
    assert memory_per_sample < dramatic_memory_threshold, \
        f"Dramatic memory regression: {memory_per_sample:.4f} MB/sample > {dramatic_memory_threshold:.4f} MB/sample"

    # Check for slow-leak memory regression (3x more memory)
    slow_leak_memory_threshold = performance_baseline['memory_per_sample'] * 3
    assert memory_per_sample < slow_leak_memory_threshold, \
        f"Slow-leak memory regression: {memory_per_sample:.4f} MB/sample > {slow_leak_memory_threshold:.4f} MB/sample"

    print(f"Memory usage: {memory_per_sample:.4f} MB/sample \
          (baseline: {performance_baseline['memory_per_sample']:.4f} MB/sample)")
