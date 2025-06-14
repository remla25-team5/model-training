import joblib
import pandas as pd
import pickle
import pytest
from pathlib import Path
import requests
import time
from sklearn.metrics import accuracy_score
from lib_ml.preprocessing import preprocess_dataset
# bandit: disable=B101  (asserts are fine in this test)

# Test Model Development:
# Model 6: Model quality is sufficient on all important data slices.


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


def test_model_performance_on_slices(raw_dataset, load_model, load_vectorizer):
    """
    Test model quality on positive and negative review slices.
    """
    # Preprocess full dataset
    corpus, labels = preprocess_dataset(raw_dataset)
    df = pd.DataFrame({
        "Review": corpus,
        "Liked": labels
    })

    cv = load_vectorizer
    classifier = load_model

    # Evaluate on positive slice
    positive_reviews = df[df['Liked'] == 1].reset_index(drop=True)
    assert len(positive_reviews) > 20, "Not enough positive reviews in test set"
    pos_features = cv.transform(positive_reviews["Review"]).toarray()
    pos_preds = classifier.predict(pos_features)
    pos_accuracy = accuracy_score(positive_reviews["Liked"], pos_preds)

    # Evaluate on negative slice
    negative_reviews = df[df['Liked'] == 0].reset_index(drop=True)
    assert len(negative_reviews) > 20, "Not enough negative reviews in test set"
    neg_features = cv.transform(negative_reviews["Review"]).toarray()
    neg_preds = classifier.predict(neg_features)
    neg_accuracy = accuracy_score(negative_reviews["Liked"], neg_preds)

    # Assertions
    assert pos_accuracy > 0.6, f"Positive review accuracy too low: {pos_accuracy}"
    assert neg_accuracy > 0.6, f"Negative review accuracy too low: {neg_accuracy}"
