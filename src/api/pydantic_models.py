from pydantic import BaseModel
from typing import Dict


class PredictionRequest(BaseModel):
    features: Dict[str, float]


class PredictionResponse(BaseModel):
    is_high_risk: int
    risk_probability: float
