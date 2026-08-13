# Student Performance Prediction — End-to-End Machine Learning Project

An end-to-end machine learning project that predicts a student's **math score** from demographic and academic background features. The project covers the full lifecycle: exploratory data analysis, a modular training pipeline, model selection across multiple regression algorithms, a Flask web interface for inference, and deployment configuration for AWS Elastic Beanstalk.

---

## Problem Statement

Given a student's gender, ethnic background, parental level of education, lunch type, test preparation course status, reading score, and writing score — **predict their math score**.

This is framed as a supervised regression problem, evaluated using the **R² score**.

---

## Dataset

The dataset (`notebook/data/data.csv`) contains 1000 student records with the following fields:

| Feature | Type | Description |
|---|---|---|
| `gender` | Categorical | Student's gender |
| `race_ethnicity` | Categorical | Ethnic group (group A–E) |
| `parental_level_of_education` | Categorical | Highest education level of parents |
| `lunch` | Categorical | Standard or free/reduced |
| `test_preparation_course` | Categorical | Completed or none |
| `reading_score` | Numerical | Score out of 100 |
| `writing_score` | Numerical | Score out of 100 |
| `math_score` | Numerical | **Target variable** |

---

## Repository Structure

```
Project_Practice/
│
├── .ebextensions/
│   └── python.config              # AWS Elastic Beanstalk WSGI configuration
│
├── artifacts/                     # Generated outputs from the training pipeline
│   ├── data.csv                   # Raw data copy
│   ├── train.csv                  # Training split
│   ├── test.csv                   # Test split
│   ├── preprocessor.pkl           # Serialised ColumnTransformer
│   └── model.pkl                  # Best-performing trained model
│
├── notebook/
│   ├── data/data.csv              # Source dataset
│   ├── EDA_MODEL.ipynb            # Exploratory data analysis
│   └── MODEL_TRAINING.ipynb       # Model experimentation
│
├── src/
│   ├── components/
│   │   ├── data_ingestion.py      # Reads raw data, performs train/test split
│   │   ├── data_transformation.py # Builds and applies the preprocessing pipeline
│   │   └── model_trainer.py       # Trains candidate models, selects the best
│   │
│   ├── pipeline/
│   │   ├── train_pipeline.py      # Orchestrates the training sequence
│   │   └── predict_pipeline.py    # Loads artifacts, serves predictions
│   │
│   ├── exception.py               # Custom exception with file/line traceback
│   ├── logger.py                  # Timestamped logging configuration
│   └── utils.py                   # Object serialisation and model evaluation
│
├── templates/
│   ├── index.html                 # Landing page
│   └── home.html                  # Prediction form and result display
│
├── app.py                         # Flask application
├── application.py                 # Deployment entry point
├── requirements.txt               # Dependencies
├── setup.py                       # Package configuration
└── README.md
```

---

## Architecture

The project follows a **modular component-based design**. Each stage is an independent, reusable module with its own configuration class, communicating with the next stage only through returned file paths and arrays — never through shared global state.

```
data.csv
   │
   ▼
┌─────────────────────┐
│   Data Ingestion    │  Read CSV → 80/20 split → write artifacts/
└─────────────────────┘
   │  train.csv, test.csv
   ▼
┌─────────────────────┐
│ Data Transformation │  Impute → encode → scale → save preprocessor.pkl
└─────────────────────┘
   │  train_arr, test_arr
   ▼
┌─────────────────────┐
│   Model Trainer     │  Train candidates → select best → save model.pkl
└─────────────────────┘
   │
   ▼
┌─────────────────────┐
│  Predict Pipeline   │  Load artifacts → transform input → predict
└─────────────────────┘
   │
   ▼
   Flask web interface
```

Every component pairs a `@dataclass` **config class** (holding output paths) with a **logic class** (doing the work). Keeping paths in one place means a directory rename is a one-line change rather than a hunt through the codebase.

---

## Pipeline Components

### 1. Data Ingestion — `src/components/data_ingestion.py`

Reads the raw CSV into a pandas DataFrame, creates the `artifacts/` directory, saves an untouched copy of the raw data, then performs an 80/20 train-test split with a fixed `random_state` for reproducibility. Returns the paths of the train and test files for the next stage.

### 2. Data Transformation — `src/components/data_transformation.py`

Builds a `ColumnTransformer` that applies two separate pipelines:

**Numerical** (`reading_score`, `writing_score`)
- `SimpleImputer(strategy='median')` — robust to outliers
- `StandardScaler()` — zero mean, unit variance

**Categorical** (`gender`, `race_ethnicity`, `parental_level_of_education`, `lunch`, `test_preparation_course`)
- `SimpleImputer(strategy='most_frequent')`
- `OneHotEncoder()` — converts categories to binary indicator columns
- `StandardScaler()` — brings encoded features onto a comparable scale

The fitted preprocessor is serialised to `artifacts/preprocessor.pkl` so that identical transformations are applied at inference time. This is critical: a model trained on scaled data must receive scaled data in production, using the *same* fitted parameters.

The stage returns transformed feature arrays with the target column concatenated as the final column.

### 3. Model Trainer — `src/components/model_trainer.py`

Splits the incoming arrays back into features and target, then trains multiple regression algorithms and compares their **R² score** on the held-out test set. Candidate models span linear, tree-based, and gradient-boosting families — including Random Forest, CatBoost, and XGBoost — so that no single algorithm family is assumed to be best for this data.

The highest-scoring model is serialised to `artifacts/model.pkl`. A minimum performance threshold guards against saving a model that fails to beat a trivial baseline.

### 4. Predict Pipeline — `src/pipeline/predict_pipeline.py`

Two classes handle inference:

- **`CustomData`** — accepts raw form inputs from the web interface and converts them into a single-row DataFrame with the exact column names the preprocessor expects.
- **`PredictPipeline`** — loads `preprocessor.pkl` and `model.pkl`, applies the transformation, and returns the predicted math score.

---

## Supporting Modules

### `src/exception.py`

Defines `CustomException`, which wraps any raised error and enriches it using `sys.exc_info()` to extract the **file name and line number** where the failure originated. Instead of a bare `ZeroDivisionError`, you get:

```
Error occurred in python script name [src/components/data_ingestion.py]
line number [26] error message [No such file or directory]
```

Every component wraps its logic in `try/except` and re-raises through this class, so failures anywhere in the pipeline are traceable to their exact source.

### `src/logger.py`

Configures Python's `logging` module to write timestamped log files into a `logs/` directory, with each run producing its own file. Components log entry and exit of each stage, making it possible to reconstruct what happened during a failed run without a debugger.

### `src/utils.py`

- **`save_object(file_path, obj)`** — serialises Python objects to disk with `dill`, creating parent directories as needed. Used for both the preprocessor and the model.
- **`load_object(file_path)`** — the inverse, used by the prediction pipeline.
- **`evaluate_model(...)`** — trains each candidate model, computes R² on train and test sets, and returns a report dictionary mapping model name to score.

---

## Web Application

A Flask app (`app.py`) provides the user-facing interface:

| Route | Method | Purpose |
|---|---|---|
| `/` | GET | Landing page (`index.html`) |
| `/predictdata` | GET | Renders the input form (`home.html`) |
| `/predictdata` | POST | Collects form data, runs prediction, returns result |

Submitted form values are passed into `CustomData`, converted to a DataFrame, and pushed through `PredictPipeline`. The predicted math score is rendered back into the same page.

---


## Installation

**Prerequisites:** Python 3.13, and Git.

**1. Clone the repository**

```bash
git clone https://github.com/shubhbagaria/Project_Practice.git
cd Project_Practice
```

**2. Create and activate a virtual environment**

```bash
python -m venv myenv

# Windows
myenv\Scripts\activate

# macOS / Linux
source myenv/bin/activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

The final line of `requirements.txt` is `-e .`, which triggers `setup.py` and installs the project itself in **editable mode**. This registers `src` as an importable package, so statements like `from src.components.data_ingestion import DataIngestion` resolve from any working directory — and any code edit takes effect immediately without reinstalling.

---

## Usage

### Train the model

Run the full training pipeline from the project root:

```bash
python src/pipeline/train_pipeline.py
```

This regenerates everything in `artifacts/` — the data splits, the fitted preprocessor, and the best model — and prints the R² score of the selected model.

### Run the web application

```bash
python app.py
```

Then open `http://127.0.0.1:5000/predictdata` in a browser, fill in the student details, and submit to receive a predicted math score.

---


## Key Concepts Demonstrated

**Modular design over notebooks.** Exploratory work lives in `notebook/`, but all production logic is refactored into importable, testable modules under `src/`.

**Configuration separated from logic.** Every component uses a `@dataclass` config holding its output paths, keeping *what a component uses* distinct from *what it does*.

**Reproducibility.** Fixed random seeds, versioned dependencies, and serialised preprocessing objects mean the same inputs produce the same outputs on any machine.

**Train/inference consistency.** Saving the fitted preprocessor — not just the model — guarantees that production data is transformed exactly as training data was.

**Traceable failures.** A custom exception class surfaces the file and line of every error, and structured logging records the path a run took before it failed.

**Cross-platform paths.** All file paths are built with `os.path.join()` rather than hardcoded separators, so the project runs unchanged on Windows and Linux.
