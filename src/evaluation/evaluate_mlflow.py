
"""evaluate_mlflow.py
Evaluate a saved model and log evaluation metrics to MLflow.
Usage:
    python src/evaluation/evaluate_mlflow.py --model-path outputs/models/model.joblib --test data/processed/test.csv --out outputs/results/
"""
import argparse, os, json
import pandas as pd, joblib
from sklearn.metrics import mean_squared_error, r2_score
import mlflow

def main(args):
    model = joblib.load(args.model_path)
    df = pd.read_csv(args.test)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    preds = model.predict(X)
    mse = mean_squared_error(y, preds)
    r2 = r2_score(y, preds)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, 'metrics.json'), 'w') as f:
        json.dump({'mse': mse, 'r2': r2}, f)
    # Log to MLflow under a new run
    mlflow.set_experiment(args.experiment or 'flight_price_eval')
    with mlflow.start_run(run_name='evaluation') as run:
        mlflow.log_metric('mse', float(mse))
        mlflow.log_metric('r2', float(r2))
    print('Saved metrics to', os.path.join(args.out, 'metrics.json'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--test', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--experiment', default=None)
    args = parser.parse_args()
    main(args)
