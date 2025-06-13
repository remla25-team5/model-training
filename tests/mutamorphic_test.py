import pandas as pd
import pytest
import requests
import time
from lib_ml.preprocessing import preprocess_dataset
from model_training.modeling.train import gaussiannb_classify
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer
# bandit: disable=B101  (asserts are fine in this test)


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


def test_mutamorphic_with_synonym_replacement(raw_dataset):
    """
    Mutamorphic test with synonym replacement to check model consistency.
    """
    # Preprocess the dataset
    corpus, labels = preprocess_dataset(raw_dataset)
    cv = CountVectorizer(max_features=1420)
    features = cv.fit_transform(corpus).toarray()

    # Train the model
    _, model = gaussiannb_classify(features, labels, cv_folds=5, random_state=42)

    # STAGE 1: Automatic test input generation
    # Define test sentences
    test_sentences = [
        'This restaurant is amazing',
        'The food was terrible',
        'I love the ambiance',
        'Service was slow',
        'The staff was friendly'
    ]

    # Define synonym mapping for automatic test input generation
    synonym_map = {
        'amazing': 'fantastic',
        'terrible': 'awful',
        'love': 'adore',
        'slow': 'sluggish',
        'friendly': 'welcoming'
    }

    mutant_map = {
        'fantastic': ['incredible', 'wonderful', 'marvelous'],
        'awful': ['dreadful', 'horrible', 'appalling'],
        'adore': ['cherish', 'appreciate', 'treasure'],
        'sluggish': ['lazy', 'unhurried', 'bad'],
        'welcoming': ['hospitable', 'warm', 'accommodating']
    }

    # Generate mutated sentences using synonyms
    mutated_sentences = []
    for i, sentence in enumerate(test_sentences):
        words = sentence.split()
        mutated_sentence = ' '.join([synonym_map.get(word, word) for word in words])
        mutated_sentences.append(mutated_sentence)
        print(f"Original: {sentence} -> Mutated: {mutated_sentence}")

        original_feature = cv.transform([sentence]).toarray()
        mutated_feature = cv.transform([mutated_sentence]).toarray()

        # STAGE 2: Automatic test oracle generation
        # Make predictions on both original and mutated features
        original_preds = model.predict(original_feature)[0]
        mutated_preds = model.predict(mutated_feature)[0]
        print(f"Original Predictions: {original_preds}")
        print(f"Mutated Predictions: {mutated_preds}")

        if not original_preds == mutated_preds:
            # STAGE 3: Automatic inconsistency repair
            # Try all mutants for the mutated sentence
            repaired = False
            words = mutated_sentences[i].split()
            for j, word in enumerate(words):
                if word in mutant_map:
                    for mutant in mutant_map[word]:
                        words[j] = mutant
                        repaired_sentence = ' '.join(words)
                        repaired_feature = cv.transform([repaired_sentence]).toarray()
                        repaired_preds = model.predict(repaired_feature)[0]
                        print(f"Repaired Sentence: {repaired_sentence} -> Prediction: {repaired_preds}")

                        if repaired_preds == original_preds:
                            print(f"Found consistent mutant: {repaired_sentence}")
                            mutated_sentences[i] = repaired_sentence
                            repaired = True
                            break

            if not repaired:
                print(f"No consistent mutant found for: {mutated_sentences[i]}")
                assert False, f"Inconsistency found between original and mutated predictions\
                      for: {mutated_sentences[i]}"
