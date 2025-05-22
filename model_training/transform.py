import pickle
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from lib_ml.preprocessing import preprocess_dataset

def main(
    input_path: Path = Path(__file__).parent.parent / "data/raw/a1_RestaurantReviews_HistoricDump.tsv",
    vectorizer_path: Path = Path(__file__).parent.parent / "models/c1_BoW_Sentiment_Model.pkl",
    data_path: Path = Path(__file__).parent.parent / "data/processed/X.pkl",
    labels_path: Path = Path(__file__).parent.parent / "data/processed/y.pkl",
):
    dataset = pd.read_csv(input_path, delimiter='\t', quoting=3)
    corpus, labels = preprocess_dataset(dataset)

    cv = CountVectorizer(max_features=1420)
    data = cv.fit_transform(corpus).toarray()

    # Ensure output dirs exist
    vectorizer_path.parent.mkdir(parents=True, exist_ok=True)
    data_path.parent.mkdir(parents=True, exist_ok=True)

    # Save BoW model
    with open(vectorizer_path, "wb") as f:
        pickle.dump(cv, f)

    # Save data and labels
    with open(data_path, "wb") as f:
        pickle.dump(data, f)
    with open(labels_path, "wb") as f:
        pickle.dump(labels, f)

    print(f"Saved BoW model to {vectorizer_path}")
    print(f"Saved features to {data_path}")
    print(f"Saved labels to {labels_path}")

if __name__ == "__main__":
    main()
