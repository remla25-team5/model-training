import pandas as pd
import pytest
from pathlib import Path
import requests
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
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
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    time.sleep(delay * (2 ** attempt))
                else:
                    raise
    
    # Load and return the dataset
    dataset = pd.read_csv(file_path, delimiter='\t', quoting=3)
    return dataset


def test_model_performance_on_slices(raw_dataset):
    """
    Test model quality on positive and negative review slices.
    """
    # Preprocess full dataset
    full_corpus, full_labels = preprocess_dataset(raw_dataset)

    # Split into train and test sets
    X_train_texts, X_test_texts, y_train, y_test = train_test_split(
        full_corpus, full_labels, test_size=0.2, random_state=42, stratify=full_labels
    )

    # Fit CountVectorizer on training texts
    cv = CountVectorizer(max_features=1420)
    X_train = cv.fit_transform(X_train_texts).toarray()
    X_test = cv.transform(X_test_texts).toarray()

    # Train classifier
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)

    # Build test set DataFrame for slicing
    test_df = pd.DataFrame({
        "Review": X_test_texts,
        "Liked": y_test
    })

    # Evaluate on positive slice
    positive_reviews = test_df[test_df['Liked'] == 1].reset_index(drop=True)
    assert len(positive_reviews) > 20, "Not enough positive reviews in test set"
    pos_features = cv.transform(positive_reviews["Review"]).toarray()
    pos_preds = classifier.predict(pos_features)
    pos_accuracy = accuracy_score(positive_reviews["Liked"], pos_preds)

    # Evaluate on negative slice
    negative_reviews = test_df[test_df['Liked'] == 0].reset_index(drop=True)
    assert len(negative_reviews) > 20, "Not enough negative reviews in test set"
    neg_features = cv.transform(negative_reviews["Review"]).toarray()
    neg_preds = classifier.predict(neg_features)
    neg_accuracy = accuracy_score(negative_reviews["Liked"], neg_preds)

    # Assertions
    assert pos_accuracy > 0.6, f"Positive review accuracy too low: {pos_accuracy}"
    assert neg_accuracy > 0.6, f"Negative review accuracy too low: {neg_accuracy}"
