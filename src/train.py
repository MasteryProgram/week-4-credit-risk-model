import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.utils.class_weight import compute_sample_weight

from src.data_processing import run_pipeline, split_data
TRUSTED_SKLEARN_TYPES = ['sklearn.tree._tree.Tree']

MLFLOW_TRACKING_URI = 'sqlite:///mlflow.db'
MLFLOW_EXPERIMENT_NAME = 'Credit_Risk_Modeling'
REGISTERED_MODEL_NAME = 'credit_risk_model'

MODEL_DIR = Path('models')
MODEL_DIR.mkdir(exist_ok=True)
PROCESSED_DATA_PATH = Path('data/processed/customer_features.csv')
RAW_DATA_PATH = Path('data/raw/alternate_data.csv')
FEATURE_LIST_PATH = MODEL_DIR / 'feature_columns.json'
BEST_MODEL_PATH = MODEL_DIR / 'best_model.joblib'


def load_data() -> pd.DataFrame:
    if not PROCESSED_DATA_PATH.exists():
        run_pipeline(str(RAW_DATA_PATH), str(PROCESSED_DATA_PATH.parent))
    return pd.read_csv(PROCESSED_DATA_PATH)


def evaluate_model(y_true, y_pred, y_prob) -> dict:
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_prob),
        'pr_auc': average_precision_score(y_true, y_prob),
    }


def tune_model(name, estimator, param_grid, X_train, y_train):
    search = GridSearchCV(estimator, param_grid, scoring='roc_auc', cv=3, n_jobs=-1)
    if name == 'GradientBoosting':
        # No class_weight parameter in sklearn's GB, so weight samples instead
        weights = compute_sample_weight(class_weight='balanced', y=y_train)
        search.fit(X_train, y_train, sample_weight=weights)
    else:
        search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_


def get_candidates() -> dict:
    return {
        'LogisticRegression': (
            LogisticRegression(max_iter=1000, random_state=42, solver='liblinear'),
            {'C': [0.01, 0.1, 1.0], 'penalty': ['l2'], 'class_weight': [None, 'balanced']},
        ),
        'RandomForest': (
            RandomForestClassifier(random_state=42),
            {'n_estimators': [100, 200], 'max_depth': [None, 8], 'class_weight': [None, 'balanced']},
        ),
        'GradientBoosting': (
            GradientBoostingClassifier(random_state=42),
            {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.05, 0.1], 'max_depth': [2, 3, 4]},
        ),
    }


def train_models() -> dict:
    df = load_data()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    best_val_auc, best_name, best_model, best_model_uri = -1.0, None, None, None
    results = {}
    example = X_train.iloc[:5]

    for name, (estimator, grid) in get_candidates().items():
        with mlflow.start_run(run_name=name) as run:
            model, params = tune_model(name, estimator, grid, X_train, y_train)

            val_metrics = evaluate_model(y_val, model.predict(X_val), model.predict_proba(X_val)[:, 1])
            test_pred = model.predict(X_test)
            test_metrics = evaluate_model(y_test, test_pred, model.predict_proba(X_test)[:, 1])

            mlflow.log_params(params)
            mlflow.log_metrics({f'val_{k}': v for k, v in val_metrics.items()})
            mlflow.log_metrics({f'test_{k}': v for k, v in test_metrics.items()})

            model_info = mlflow.sklearn.log_model(
                model,
                'model',
                signature=infer_signature(example, model.predict(example)),
                input_example=example,
                skops_trusted_types=TRUSTED_SKLEARN_TYPES,
            )

            results[name] = {
                'params': params,
                'val_metrics': val_metrics,
                'test_metrics': test_metrics,
                'report': classification_report(y_test, test_pred),
                'run_id': run.info.run_id,
            }

            # Winner is chosen on validation ROC-AUC; test set is only reported
            if val_metrics['roc_auc'] > best_val_auc:
                best_val_auc, best_name = val_metrics['roc_auc'], name
                best_model, best_model_uri = model, model_info.model_uri

    registered = mlflow.register_model(model_uri=best_model_uri, name=REGISTERED_MODEL_NAME)
    print(f'Registered {REGISTERED_MODEL_NAME} v{registered.version} '
          f'({best_name}, val ROC-AUC={best_val_auc:.4f})')

    joblib.dump(best_model, BEST_MODEL_PATH)
    FEATURE_LIST_PATH.write_text(json.dumps(list(X_train.columns), indent=2), encoding='utf-8')
    return results


if __name__ == '__main__':
    output = train_models()
    print('\nTest-set metrics:')
    for model_name, meta in output.items():
        tm = meta['test_metrics']
        print(f"- {model_name}: ROC-AUC={tm['roc_auc']:.4f} PR-AUC={tm['pr_auc']:.4f} "
              f"Recall={tm['recall']:.4f} Precision={tm['precision']:.4f} F1={tm['f1']:.4f}")
    print(f'Best model saved to {BEST_MODEL_PATH}')