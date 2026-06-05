import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split

from src.data_processing import run_pipeline

MLFLOW_EXPERIMENT_NAME = 'Credit_Risk_Modeling'
MODEL_DIR = Path('models')
MODEL_DIR.mkdir(exist_ok=True)

PROCESSED_DATA_PATH = Path('data/processed/customer_features.csv')
RAW_DATA_PATH = Path('data/raw/alternate_data.csv')
FEATURE_LIST_PATH = MODEL_DIR / 'feature_columns.json'
BEST_MODEL_PATH = MODEL_DIR / 'best_model.joblib'


def load_data():
    if not PROCESSED_DATA_PATH.exists():
        run_pipeline(str(RAW_DATA_PATH), str(PROCESSED_DATA_PATH.parent))
    df = pd.read_csv(PROCESSED_DATA_PATH)
    X = df.drop(['is_high_risk', 'CustomerId'], axis=1, errors='ignore')
    y = df['is_high_risk']
    return X, y


def evaluate_model(y_true, y_pred, y_prob=None):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_prob) if y_prob is not None else None,
    }


def tune_model(estimator, param_grid, X_train, y_train):
    if not param_grid:
        estimator.fit(X_train, y_train)
        return estimator, estimator.get_params()
    search = GridSearchCV(estimator, param_grid, scoring='roc_auc', cv=3, n_jobs=-1, verbose=0)
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_


def train_models():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    candidates = {
        'LogisticRegression': (
            LogisticRegression(max_iter=1000, random_state=42, solver='liblinear'),
            {'C': [0.01, 0.1, 1.0], 'penalty': ['l2']},
        ),
        'RandomForest': (
            RandomForestClassifier(random_state=42),
            {'n_estimators': [100, 200], 'max_depth': [None, 8]},
        ),
        'GradientBoosting': (
            GradientBoostingClassifier(random_state=42),
            {'n_estimators': [100], 'learning_rate': [0.1], 'max_depth': [3]},
        ),
    }

    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    best_score = -1.0
    best_overall_model = None
    results = {}

    for name, (estimator, grid) in candidates.items():
        with mlflow.start_run(run_name=name):
            best_model, best_params = tune_model(estimator, grid, X_train, y_train)
            y_pred = best_model.predict(X_test)
            y_prob = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, 'predict_proba') else None
            metrics = evaluate_model(y_test, y_pred, y_prob)
            mlflow.log_params(best_params)
            for key, value in metrics.items():
                if value is not None:
                    mlflow.log_metric(key, value)
            mlflow.sklearn.log_model(best_model, 'model')
            results[name] = {
                'model': best_model,
                'params': best_params,
                'metrics': metrics,
                'report': classification_report(y_test, y_pred),
            }
            if metrics['roc_auc'] is not None and metrics['roc_auc'] > best_score:
                best_score = metrics['roc_auc']
                best_overall_model = best_model

    if best_overall_model is None:
        raise RuntimeError('No model was trained successfully.')

    joblib.dump(best_overall_model, BEST_MODEL_PATH)
    FEATURE_LIST_PATH.write_text(json.dumps(list(X.columns), indent=2), encoding='utf-8')
    return results


if __name__ == '__main__':
    print('Training models...')
    results = train_models()
    print('\nTraining complete. Models trained:')
    for name, meta in results.items():
        print(f'- {name}: ROC-AUC={meta["metrics"]["roc_auc"]:.4f}, F1={meta["metrics"]["f1"]:.4f}')
    print(f'Best model saved to {BEST_MODEL_PATH}')
