import pickle
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

from lib_ml.preprocessing import preprocess_dataset


def main(
    input_path: Path = Path(__file__).parent.parent / "data/raw/a1_RestaurantReviews_HistoricDump.tsv",
    vectorizer_path: Path = Path(__file__).parent.parent / "models/c1_BoW_Sentiment_Model.pkl",
    train_data_path: Path = Path(__file__).parent.parent / "data/processed/X_train.pkl",
    train_labels_path: Path = Path(__file__).parent.parent / "data/processed/y_train.pkl",
    test_data_path: Path = Path(__file__).parent.parent / "data/processed/X_test.pkl",
    test_labels_path: Path = Path(__file__).parent.parent / "data/processed/y_test.pkl",
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Processes raw data, fits a BoW vectorizer, and creates train/test splits.

    Saves the fitted vectorizer and all data splits to specified output paths.
    """
    # Load and preprocess
    dataset = pd.read_csv(input_path, delimiter='\t', quoting=3)
    corpus, labels = preprocess_dataset(dataset)

    # Fit vectorizer
    cv = CountVectorizer(max_features=1420)
    data = cv.fit_transform(corpus).toarray()

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=test_size, random_state=random_state, stratify=labels
    )

    # Ensure output dirs exist
    vectorizer_path.parent.mkdir(parents=True, exist_ok=True)
    train_data_path.parent.mkdir(parents=True, exist_ok=True)

    # Save vectorizer
    with open(vectorizer_path, "wb") as f:
        pickle.dump(cv, f)

    # Save splits
    with open(train_data_path, "wb") as f:
        pickle.dump(X_train, f)
    with open(train_labels_path, "wb") as f:
        pickle.dump(y_train, f)
    with open(test_data_path, "wb") as f:
        pickle.dump(X_test, f)
    with open(test_labels_path, "wb") as f:
        pickle.dump(y_test, f)

    print(f"Saved BoW model to {vectorizer_path}")
    print(f"Saved training features to {train_data_path}")
    print(f"Saved training labels to {train_labels_path}")
    print(f"Saved test features to {test_data_path}")
    print(f"Saved test labels to {test_labels_path}")


if __name__ == "__main__":
    main()
