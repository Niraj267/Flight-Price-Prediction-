
"""clean_data.py
Data cleaning & feature engineering for flight price dataset.
Usage:
    python src/preprocessing/clean_data.py --input data/raw/flights.csv --output data/processed/processed.csv
"""
import argparse, os, pandas as pd, numpy as np

def parse_duration(x):
    # Example helper: '2h 50m' -> minutes
    try:
        parts = x.strip().split()
        total = 0
        for p in parts:
            if p.endswith('h'):
                total += int(p[:-1]) * 60
            elif p.endswith('m'):
                total += int(p[:-1])
        return total
    except Exception:
        return np.nan

def feature_engineer(df):
    df = df.copy()
    # Example featurizations used commonly in flight datasets - adjust to your columns
    if 'Date_of_Journey' in df.columns:
        df['Date_of_Journey'] = pd.to_datetime(df['Date_of_Journey'], errors='coerce')
        df['journey_day'] = df['Date_of_Journey'].dt.day
        df['journey_month'] = df['Date_of_Journey'].dt.month
        df['journey_weekday'] = df['Date_of_Journey'].dt.weekday
    if 'Dep_Time' in df.columns:
        df['Dep_Time'] = pd.to_datetime(df['Dep_Time'], format='%H:%M', errors='coerce')
        df['dep_hour'] = df['Dep_Time'].dt.hour
        df['dep_min'] = df['Dep_Time'].dt.minute
    if 'Arrival_Time' in df.columns:
        df['Arrival_Time'] = pd.to_datetime(df['Arrival_Time'], format='%H:%M', errors='coerce')
        df['arr_hour'] = df['Arrival_Time'].dt.hour
        df['arr_min'] = df['Arrival_Time'].dt.minute
    if 'Duration' in df.columns:
        df['duration_mins'] = df['Duration'].apply(parse_duration)
    # Handle total stops
    if 'Total_Stops' in df.columns:
        df['Total_Stops'] = df['Total_Stops'].replace('non-stop', '0 stop').str.extract('(\d+)').astype(float)
    # Normalize categorical strings
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.strip().str.lower()
    # Fill missing numeric with median
    for col in df.select_dtypes(include=['float','int']).columns:
        df[col] = df[col].fillna(df[col].median())
    # Fill missing categorical with mode
    for col in df.select_dtypes(include=['object','category']).columns:
        df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else 'unknown')
    return df

def main(args):
    inp = args.input
    out = args.output
    if os.path.isdir(inp):
        files = [f for f in os.listdir(inp) if f.lower().endswith('.csv')]
        if not files:
            raise FileNotFoundError('No CSV files found in input directory')
        inp = os.path.join(inp, files[0])
    df = pd.read_csv(inp)
    df_clean = feature_engineer(df)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df_clean.to_csv(out, index=False)
    print('Saved processed data to', out)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    main(args)
