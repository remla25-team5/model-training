![test coverage](https://img.shields.io/badge/test%20coverage-98%25-green.svg)
![pylint score](https://img.shields.io/badge/pylint%20score-10.0-green.svg)
![flake8](https://img.shields.io/badge/flake8-0%20issues-brightgreen.svg)
![bandit](https://img.shields.io/badge/bandit-0%20issues-brightgreen.svg)
![ml score](https://img.shields.io/badge/ml%20score-1-blue.svg)

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

### Running the tests

This project uses `pytest` for testing, and integrates several code quality tools including `pylint`, `flake8`, and `bandit`. Code coverage is tracked using `pytest-cov`.

**Note**: Before running any tests, ensure you have the model and BoW vectorizer files in the `models/` folder. You can download them by running `model_training/dataset.py`, then `model_training/transform.py`, and finally `model_training/modeling/train.py` to train the model and save it in the `models` folder.

#### Running Unit Tests with pytest

To run all tests in the tests folder:

```bash
pytest tests/
```

To run a specific test file:

```bash
pytest tests/test_features_data.py
```

#### Mutamorphic Testing with Automatic Repair

The project includes mutamorphic tests that automatically detect and repair inconsistencies in model behavior. These tests check if model predictions remain consistent across different mutations of the same test input data, such as replacing words with synonyms.

```bash
pytest tests/mutamorphic_test.py -v
```

Mutamorphic testing works by:
1. Generating the test input data.
2. Testing if replacing words with synonyms leads to the same model predictions.
3. If predictions differ, the test fails and the code attempts to repair the issue by trying other synonyms to automatically repair consistencies.

#### Checking Code Coverage

You can run tests with coverage reporting:

```bash
uv run pytest --doctest-modules --junitxml=junit/test-results.xml --cov=model_training --cov-report=json --cov-report=term
```

This generates:
- A terminal coverage report
- A JSON coverage report in `coverage.json`
- JUnit test results in `junit/test-results.xml`

For a more detailed HTML coverage report:

```bash
uv run pytest --doctest-modules --cov=model_training --cov-report=html
```

This creates an HTML report in the `htmlcov` folder, which you can open in a browser to view detailed coverage information.

#### Code Quality Tools

**Flake8**: Check for PEP8 compliance and other issues:

```bash
flake8 --config=.flake8
```

**Pylint**: Perform a more in-depth analysis of your code, checking for programming errors, enforcing coding standards, and looking for code smells. A custom pylint plugin is also added (pylint_plugins/random_state_checker.py), which checks the followinn smell: https://hynn01.github.io/ml-smells/posts/codesmells/14-randomness-uncontrolled/. 

```bash
pylint ./
```

**Bandit**: Scan your code for common security vulnerabilities.

```bash
bandit -c bandit.yaml -r .\ 
```

#### ML Test Score

The testing framework also calculates an ML Test Score based on test results across different categories:
- Features
- Monitoring
- ML Infrastructure
- Model Development

These scores are displayed in the test output and contribute to the ML Score badge in this README.


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

