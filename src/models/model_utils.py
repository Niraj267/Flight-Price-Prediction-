
"""model_utils.py
Helpers to load and save models and perform inference
"""
import joblib, os
def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)

def load_model(path):
    return joblib.load(path)
