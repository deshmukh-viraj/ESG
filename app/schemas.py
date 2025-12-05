from pydantic import BaseModel, Field
from typing import List


class PredictionRequest(BaseModel):
    CompanyID: int
    Year: int
    Revenue: float
    CarbonEmissions: float
    WaterUsage: float
    EnergyConsumption: float
    ProfitMargin: float
    ESG_OVERALL_PREV: float = Field(..., description="Previous year's ESG overall score")

class PredictionResponse(BaseModel):
    prediction: float

class BatchPredictionRespinse(BaseModel):
    predictions: List[float]
    total: int