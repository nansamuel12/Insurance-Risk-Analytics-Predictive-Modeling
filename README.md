# Insurance Risk Analytics Predictive Modeling

This repository hosts a predictive modeling project for insurance risk analytics. It demonstrates best practices in software engineering for data science, including version control with Git, data version control with DVC, and reproducible pipelines.

## 📂 Project Structure

The project follows a clean, modular structure:

```
├── data/
│   ├── raw/            # Raw immutable data (tracked by DVC)
│   └── processed/      # Processed data (tracked by DVC)
├── docs/               # Project documentation
├── reports/            # Generated reports and visualizations
│   ├── figures/        # EDA visualizations
│   ├── ab_hypothesis_tests.json    # Statistical test results
│   └── ab_hypothesis_report.txt    # Detailed hypothesis test report
├── src/                # Source code for pipelines
│   ├── process_data.py           # Script to clean and transform data
│   ├── train_model.py            # Script to train models and calculate metrics
│   ├── eda.py                    # Exploratory data analysis
│   └── ab_hypothesis_testing.py  # A/B hypothesis testing module
├── dvc.yaml            # DVC pipeline configuration
├── dvc.lock            # DVC lockfile for reproducibility
├── metrics.json        # Model performance metrics
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Git

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/nansamuel12/Insurance-Risk-Analytics-Predictive-Modeling
   cd Insurance-Risk-Analytics-Predictive-Modeling
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Pull data and artifacts:**
   This project uses DVC to manage large files. Pull the data from the configured remote storage:
   ```bash
   dvc pull
   ```

## 🔄 Reproducible Pipeline

We use DVC to define a reproducible DAG (Directed Acyclic Graph) of steps.

**Run the full pipeline:**
```bash
dvc repro
```

**Pipeline Steps:**
1. **EDA (Exploratory Data Analysis)**: Generates visualizations and data quality reports from raw data.
2. **A/B Hypothesis Testing**: Performs comprehensive statistical hypothesis tests:
   - **Chi-Square Tests**: Tests independence between categorical variables (e.g., smoker vs sex, smoker vs region)
   - **Independent T-Tests**: Compares means between two groups (e.g., charges for smokers vs non-smokers)
   - **ANOVA Tests**: Compares means across multiple groups (e.g., charges across regions)
   - Calculates effect sizes (Cohen's d, Cramér's V, eta-squared)
   - Generates statistical reports and visualizations
3. **Statistical Modeling**: Builds and evaluates predictive models:
   - **Linear Regression**: Baseline linear model
   - **Ridge Regression**: L2 regularized linear model with hyperparameter tuning
   - **Lasso Regression**: L1 regularized linear model with feature selection
   - **Random Forest**: Ensemble tree-based model
   - **Gradient Boosting**: Advanced ensemble boosting model
   - Performs cross-validation and hyperparameter optimization
   - Generates model comparison reports and visualizations
4. **Process Data**: Reads `data/raw/data.csv`, performs transformations (e.g., doubling values), and saves to `data/processed/processed_data.csv`.
5. **Train Model**: Reads the processed data, calculates statistics (Mean, Std Dev), and saves them to `metrics.json`.

## 🛠 Implementation Journey

This project was built iteratively following DevOps best practices:

### 1. Repository Setup
- Initialized a Git repository.
- Created `task1` and `task2` branches for parallel development.
- Established a `main` branch as the source of truth.

### 2. Data Version Control (DVC) Integration
- **Goal**: Track large datasets and ensure auditability.
- **Steps**:
  - Installed DVC: `pip install dvc`
  - Initialized DVC: `dvc init`
  - Configured local storage: `dvc remote add -d localstorage /path/to/storage`
  - Tracked data: `dvc add data/raw/data.csv`

### 3. A/B Hypothesis Testing Implementation
- **Goal**: Perform rigorous statistical analysis to identify significant differences between groups in the insurance dataset.
- **Statistical Tests Implemented**:
  - **Chi-Square Test of Independence**: Analyzes relationships between categorical variables
  - **Independent Samples T-Test**: Compares means between two independent groups
  - **One-Way ANOVA**: Compares means across multiple groups
- **Features**:
  - Automated hypothesis testing with proper null/alternative hypothesis formulation
  - Effect size calculations (Cohen's d, Cramér's V, eta-squared) for practical significance
  - Statistical significance testing at α = 0.05 level
  - Comprehensive reporting with interpretations
  - Visualizations of p-values, effect sizes, and group comparisons
- **Key Findings**:
  - **Significant Result**: Smokers have significantly higher insurance charges than non-smokers (p < 0.001, Cohen's d = 2.73)
  - Sex and region show no significant impact on charges
  - Smoking status is independent of sex and region

### 4. Statistical Modeling Implementation
- **Goal**: Build predictive models to forecast insurance charges based on risk factors.
- **Models Implemented**:
  - **Linear Regression**: Baseline model using ordinary least squares
  - **Ridge Regression**: L2 regularization to prevent overfitting
  - **Lasso Regression**: L1 regularization with automatic feature selection
  - **Random Forest**: Ensemble of decision trees for non-linear relationships
  - **Gradient Boosting**: Sequential ensemble method for high accuracy
- **Features**:
  - Automated data preprocessing (encoding categorical variables, scaling features)
  - Train/test split (80/20) for unbiased evaluation
  - Grid search with cross-validation for hyperparameter tuning
  - Comprehensive metrics: R², RMSE, MAE, MAPE
  - Feature importance analysis for interpretability
  - Residual analysis for model diagnostics
  - Visualizations: model comparison, actual vs predicted, residuals
- **Key Findings**:
  - **Best Model**: Linear Regression (Test R² = -0.09, RMSE = $6,750)
  - **Challenge**: Small dataset (20 samples) leads to overfitting in complex models
  - **Overfitting Detected**: Tree-based models (Random Forest, Gradient Boosting) show perfect training performance but poor test performance
  - **Recommendation**: Collect more data for reliable predictions; Linear Regression provides most stable results

### 5. Pipeline Automation
- **Goal**: Connect data processing and modeling steps.
- **Steps**:
  - Created `src/process_data.py` and `src/train_model.py`.
  - Defined `dvc.yaml` to link inputs (dependencies) and outputs.
  - Used `dvc repro` to execute the pipeline and generate `dvc.lock`.

### 6. Continuous Improvement
- **Feature Branching**: Used branches like `feature/enhance-metrics` to add new features (e.g., standard deviation calculation) without breaking `main`.
- **Refactoring**: Moved files into `data/raw` and `docs/` folders for better organization on the `chore/repo-structure-docs` branch.

## 📜 Command Reference

Here is a list of commands used during the development of this project:

### Git Commands
- `git init`: Initialize a new repository.
- `git clone <url>`: Clone an existing repository.
- `git branch <name>`: Create a new branch.
- `git checkout -b <name>`: Create and switch to a new branch.
- `git add .`: Stage changes for commit.
- `git commit -m "message"`: Commit staged changes.
- `git merge <branch>`: Merge a branch into the current branch.
- `git push origin main`: Push changes to the remote repository.

### DVC Commands
- `dvc init`: Initialize DVC in the project.
- `dvc add <file>`: Start tracking a file with DVC.
- `dvc remote add -d <name> <path>`: Add a remote storage location.
- `dvc push`: Push tracked files to remote storage.
- `dvc pull`: Pull tracked files from remote storage.
- `dvc repro`: Reproduce the data pipeline defined in `dvc.yaml`.
- `dvc remove <file>.dvc`: Stop tracking a file.

---
*Maintained by [nansamuel12](https://github.com/nansamuel12)*
