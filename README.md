# Model-training: Restaurant Sentiment Model Training

This repository contains the ML pipeline for training the restaurant sentiment analysis model.

**Functionality:**
*   Trains a sentiment classification model on restaurant reviews.
*   Uses preprocessing steps defined in the `lib-ml` library.
*   Outputs versioned model artifacts (`.joblib`, `.pkl`).

**Releasing:**
*   Pushing a Git tag (e.g., `v1.0.0`) triggers a GitHub Action.
*   The Action creates a GitHub Release with the corresponding model artifacts attached, providing a link for the `model-service`.

**Setup:**
```bash
pip install -r requirements.txt
