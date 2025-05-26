![test coverage](https://img.shields.io/badge/test%20coverage-30%25-green.svg)
![pylint score](https://img.shields.io/badge/pylint%20score-10.0-green.svg)
![flake8](https://img.shields.io/badge/flake8-0%20issues-brightgreen.svg)
![bandit](https://img.shields.io/badge/bandit-0%20issues-brightgreen.svg)

# Model-training: Restaurant Sentiment Model Training

This repository contains the ML pipeline for training the restaurant sentiment analysis model.

**Functionality:**
*   Trains a sentiment classification model on restaurant reviews.
*   Uses preprocessing steps defined in the `lib-ml` library.
*   Outputs versioned model artifacts (`.joblib`, `.pkl`). These models can be downloaded in code using the url: `https://github.com/remla25-team5/model-training/releases/download/<TAG>/<MODEL_FILENAME.joblib_or_pkl>`

**Releasing:**
*   Pushing a Git tag (e.g., `v1.0.0`) triggers a GitHub Action.
*   The Action creates a GitHub Release with the corresponding model artifacts attached, providing a link for the `model-service`.

**Project Setup with `uv`**

This guide explains how to set up the Python development environment using [`uv`](https://github.com/astral-sh/uv), a fast Python package manager and installer.

#### Prerequisites (Windows)

- Python 3.8+
- [`uv`](https://astral.sh/uv) installed:
  ```bash
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/0.7.6/install.ps1 | iex"
    ```

```bash
uv venv
.\.venv\Scripts\Activate
uv pip install .
```

Now you have a .venv folder which contains the virtual environment (and it's activated for you), use 
```bash
  python model_training/dataset.py
```
To download from Google Cloud Bucket, this is just a single step of the entire pipeline.




# model-training

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Sentiment analysis

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         model_training and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── model_training   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes model_training a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

