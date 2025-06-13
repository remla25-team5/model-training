import joblib
import pickle
import pytest
from pathlib import Path
# bandit: disable=B101  (asserts are fine in this test)


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


def test_mutamorphic_with_synonym_replacement(load_vectorizer, load_model):
    """
    Mutamorphic test with synonym replacement to check model consistency.
    """
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

        original_feature = load_vectorizer.transform([sentence]).toarray()
        mutated_feature = load_vectorizer.transform([mutated_sentence]).toarray()

        # STAGE 2: Automatic test oracle generation
        # Make predictions on both original and mutated features
        original_preds = load_model.predict(original_feature)[0]
        mutated_preds = load_model.predict(mutated_feature)[0]
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
                        repaired_feature = load_vectorizer.transform([repaired_sentence]).toarray()
                        repaired_preds = load_model.predict(repaired_feature)[0]
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
