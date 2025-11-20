
"""train_mlflow.py
Train models and log experiments with MLflow.
Usage:
    python src/models/train_mlflow.py --data data/processed/processed.csv --experiment flight_price_exp --model xgboost
"""
import argparse, os, joblib, json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import mlflow.sklearn
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

def get_model(name, params=None):
    params = params or {}
    if name == 'xgboost':
        return xgb.XGBRegressor(**params, verbosity=0)
    if name == 'lightgbm':
        return lgb.LGBMRegressor(**params)
    if name == 'catboost':
        return CatBoostRegressor(**params, verbose=0)
    if name == 'rf':
        return RandomForestRegressor(**params)
    raise ValueError('Unknown model: ' + name)

def main(args):
    df = pd.read_csv(args.data)
    # Simple: assume last column is target. Adapt as needed.
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Basic automatic column type detection
    num_cols = X.select_dtypes(include=['int64','float64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object','category','bool']).columns.tolist()

    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('ohe', OneHotEncoder(handle_unknown='ignore', sparse=False))])
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, num_cols),
        ('cat', categorical_transformer, cat_cols)
    ])

    model = get_model(args.model)
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])

    mlflow.set_experiment(args.experiment)
    with mlflow.start_run(run_name=f"{args.model}_run") as run:
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_val)
        mse = mean_squared_error(y_val, preds)
        r2 = r2_score(y_val, preds)
        mlflow.log_param('model', args.model)
        mlflow.log_metric('mse', mse)
        mlflow.log_metric('r2', r2)
        # log model
        mlflow.sklearn.log_model(pipeline, 'model_pipeline')
        os.makedirs(args.output, exist_ok=True)
        model_path = os.path.join(args.output, f'{args.model}_pipeline.joblib')
        joblib.dump(pipeline, model_path)
        print('Saved model to', model_path)
        # write run info
        with open(os.path.join(args.output, 'last_run_info.json'), 'w') as f:
            json.dump({'run_id': run.info.run_id, 'model_path': model_path}, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='processed data CSV path')
    parser.add_argument('--experiment', default='flight_price_exp', help='MLflow experiment name')
    parser.add_argument('--model', default='xgboost', help='model to train: xgboost|lightgbm|catboost|rf')
    parser.add_argument('--output', default='outputs/models', help='folder to save model artifacts')
    args = parser.parse_args()
    main(args)
