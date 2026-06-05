from fastapi import FastAPI, HTTPException

from src.api.pydantic_models import PredictionRequest, PredictionResponse
from src.predict import load_model, predict_from_dict

app = FastAPI(title='Credit Risk Prediction API')

try:
    model, feature_columns = load_model()
except Exception as exc:
    model = None
    feature_columns = None
    print(f'Warning: model could not be loaded: {exc}')


@app.get('/health')
def health_check():
    return {'status': 'ok'}


@app.post('/predict', response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if model is None or feature_columns is None:
        raise HTTPException(status_code=500, detail='Model is not available')

    try:
        result = predict_from_dict(request.features, model=model, feature_columns=feature_columns)
        return PredictionResponse(**result)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
