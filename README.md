![test coverage](https://img.shields.io/badge/test%20coverage-0%25-green.svg)
![pylint score](https://img.shields.io/badge/pylint%20score-8.35-green.svg)

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

Try to run the command `uv --version` to check if it is installed correctly. If you see the version number, you are good to go. Otherwise, you can install `pipx` and then install uv through:

```bash
pipx install uv
```

Once you have uv installed, you can create a virtual environment and install the dependencies using the following command:

```bash
uv venv
.\.venv\Scripts\Activate
uv pip install .
```

Now you have a .venv folder which contains the virtual environment (and it's activated for you), use 
```bash
  python model_training/dataset.py
```
to download from Google Cloud Bucket, this is just a single step of the entire pipeline.

### Using DVC for ML Configuration Management

This project uses Google Drive as a remote storage for DVC. You have to set up the credentials for the remote storage. Since these credentials cannot be publicly shared for privacy reasons, you can send an email to K.Hoxha@student.tudelft.nl and request the credentials. You will also have to provide your google account email address so that the GDrive can be shared with you.

Once you have the credentials, you can set them up in your local DVC configuration. You can do this by replacing `<client-id>` and `<secret>` with the real values, and running:

```bash
dvc remote modify group5-remote gdrive_client_id '<client-id>' --local
```

```bash
dvc remote modify group5-remote gdrive_client_secret '<secret>' --local
```

Finally, you can run the following command to pull the data from the remote storage:

```bash
dvc pull
```

This will prompt you to log in to your Google account. Proceed to the page even though it is labelled as 'unsafe' and authorize DVC to access your Google Drive. Once you have authorized DVC, it will download the data from the remote storage to your local machine.

To run the pipeline, you can use the following command:

```bash
dvc repro
```

To run the experiments, you can use the following command:

```bash
dvc exp run
```

You can see the list of experiments by running:

```bash
dvc exp show
```

If you want to push any changes to the remote storage, you can run:

```bash
dvc push
```

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

