# Insurance Risk Analytics Predictive Modeling

This repository hosts a predictive modeling project for insurance risk analytics. It demonstrates best practices in software engineering for data science, including version control with Git, data version control with DVC, and reproducible pipelines.

## 📂 Project Structure

The project follows a clean, modular structure:

```
├── data/
│   ├── raw/            # Raw immutable data (tracked by DVC)
│   └── processed/      # Processed data (tracked by DVC)
├── docs/               # Project documentation
├── src/                # Source code for pipelines
│   ├── process_data.py # Script to clean and transform data
│   └── train_model.py  # Script to train models and calculate metrics
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
1. **Process Data**: Reads `data/raw/data.csv`, performs transformations (e.g., doubling values), and saves to `data/processed/processed_data.csv`.
2. **Train Model**: Reads the processed data, calculates statistics (Mean, Std Dev), and saves them to `metrics.json`.

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

### 3. Pipeline Automation
- **Goal**: Connect data processing and modeling steps.
- **Steps**:
  - Created `src/process_data.py` and `src/train_model.py`.
  - Defined `dvc.yaml` to link inputs (dependencies) and outputs.
  - Used `dvc repro` to execute the pipeline and generate `dvc.lock`.

### 4. Continuous Improvement
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
