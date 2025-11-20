
"""file_utils.py: small helpers to read/write JSON/CSV"""
import json, pandas as pd, os
def read_json(path):
    with open(path,'r') as f:
        return json.load(f)
def write_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path,'w') as f:
        json.dump(obj, f, indent=2)
def read_csv(path):
    return pd.read_csv(path)
