import json
from pathlib import Path

import joblib
import pandas as pd

MODEL_PATH = Path('models/best_model.joblib')
FEATURE_PATH = Path('models/feature_columns.json')


def load_model(model_path: Path = MODEL_PATH, feature_path: Path = FEATURE_PATH):
    model = joblib.load(model_path)
    with open(feature_path, 'r', encoding='utf-8') as f:
        feature_columns = json.load(f)
    return model, feature_columns


def predict_from_dict(feature_dict: dict, model=None, feature_columns=None):
    if model is None or feature_columns is None:
        model, feature_columns = load_model()
    missing = [c for c in feature_columns if c not in feature_dict]
    if missing:
        raise ValueError(f'Missing feature values for: {missing}')
    x = pd.DataFrame([feature_dict])[feature_columns]
    probability = model.predict_proba(x)[:, 1][0] if hasattr(model, 'predict_proba') else float(model.predict(x)[0])
    label = int(probability >= 0.5)
    return {'is_high_risk': label, 'risk_probability': float(probability)}
