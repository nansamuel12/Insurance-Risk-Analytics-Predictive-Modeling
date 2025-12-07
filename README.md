# Insurance Risk Analytics Predictive Modeling

This repository contains predictive modeling for insurance risk analytics.

## Project Structure

```
├── data/
│   ├── raw/            # Raw immutable data (DVC tracked)
│   └── processed/      # Processed data (DVC tracked)
├── docs/               # Documentation
├── src/                # Source code
│   ├── process_data.py # Data processing script
│   └── train_model.py  # Model training script
├── dvc.yaml            # DVC pipeline definition
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/nansamuel12/Insurance-Risk-Analytics-Predictive-Modeling
   cd Insurance-Risk-Analytics-Predictive-Modeling
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Pull data from DVC remote:
   ```bash
   dvc pull
   ```

## Usage

To run the full data pipeline (processing -> training):

```bash
dvc repro
```

This will:
1. Process the raw data in `data/raw/`.
2. Save processed data to `data/processed/`.
3. Train the model and save metrics to `metrics.json`.
