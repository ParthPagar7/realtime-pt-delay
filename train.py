import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib
import xgboost as xgb

def load_data(path="historical_delays.csv"):
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df.sort_values("timestamp", inplace=True)
    return df

def feature_engineer(df):
    df = df.copy()
    df['route_idx'] = df['route_id'].astype('category').cat.codes
    df['stop_idx'] = df['stop_id'].astype('category').cat.codes
    df['hour'] = df['timestamp'].dt.hour
    df['is_peak'] = df['hour'].isin([7,8,9,17,18,19]).astype(int)
    agg = (
        df.groupby(['route_id', 'stop_id'])['delay_min']
        .agg(['mean'])
        .reset_index()
        .rename(columns={'mean': 'route_stop_mean_delay'})
    )
    df = df.merge(agg, on=['route_id', 'stop_id'], how='left')
    df['route_stop_mean_delay'] = df['route_stop_mean_delay'].fillna(df['delay_min'].mean())
    features = [
        'day_of_week', 'time_of_day', 'weather_temp', 'weather_precip', 'traffic_index',
        'scheduled_minute_of_day', 'route_idx', 'stop_idx', 'hour', 'is_peak', 'route_stop_mean_delay'
    ]
    return df, features

def train_and_save(df, features, model_path="xgb_model.joblib", meta_path="meta.joblib"):
    X = df[features]
    y = df['delay_min']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        random_state=42,
        early_stopping_rounds=20
        )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
        )
    preds = model.predict(X_val)
    mae = mean_absolute_error(y_val, preds)
    print(f"Validation MAE: {mae:.3f} minutes")
    joblib.dump(model, model_path)
    joblib.dump({
        "features": features,
        "route_categories": df['route_id'].astype('category').cat.categories.tolist(),
        "stop_categories": df['stop_id'].astype('category').cat.categories.tolist()
    }, meta_path)
    print("Saved", model_path, meta_path)

if __name__ == "__main__":
    df = load_data()
    df, features = feature_engineer(df)
    train_and_save(df, features)
