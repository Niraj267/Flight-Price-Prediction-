# Flight-price-prediction-model-Advanced-MLflow

**Flight Price Prediction Model (Advanced)** — enterprise-grade ML project with MLflow experiment tracking, model registry, and ensemble modeling.

## Highlights
- Structured preprocessing and feature engineering modules
- Multiple models (XGBoost, LightGBM, CatBoost, RandomForest baseline)
- Experiment tracking with MLflow (local mlruns/ directory by default)
- Hyperparameter tuning (Optuna integration example)
- Model comparison, plotting, feature importance
- Ready for CI and deployment

## Getting started

1. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate      # macOS / Linux
venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

2. Place raw dataset(s) in `data/raw/` (CSV). The repository keeps the original uploaded notebook at `notebooks/FPP_original.ipynb` (local path was `/mnt/data/FPP.ipynb`).

3. Run preprocessing:
```bash
python src/preprocessing/clean_data.py --input data/raw/your.csv --output data/processed/processed.csv
```

4. Start MLflow UI (in the project root) to view experiments and models:
```bash
mlflow ui --backend-store-uri file:./mlruns --port 5000
# Then open http://localhost:5000
```

5. Train with MLflow logging (example):
```bash
python src/models/train_mlflow.py --data data/processed/processed.csv --experiment flight_price_exp --model xgboost
```

6. Evaluate:
```bash
python src/evaluation/evaluate_mlflow.py --model-path outputs/models/best_model.joblib --test data/processed/test.csv --out outputs/results/
```

## Project structure
```
Flight-price-prediction-model-Advanced-MLflow/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── FPP_original.ipynb
├── src/
│   ├── preprocessing/
│   │   └── clean_data.py
│   ├── models/
│   │   ├── train_mlflow.py
│   │   └── model_utils.py
│   ├── evaluation/
│   │   └── evaluate_mlflow.py
│   ├── pipeline/
│   │   └── price_pipeline.py
│   └── utils/
│       ├── logger.py
│       └── file_utils.py
├── data/
│   ├── raw/
│   └── processed/
├── experiments/
│   └── experiment_log.csv
└── outputs/
    ├── models/
    ├── plots/
    └── results/
```

---
Generated on 2025-11-20 15:38:39 UTC.
