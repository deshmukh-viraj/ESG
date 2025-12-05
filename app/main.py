# main.py
import io
import logging
import pandas as pd
import numpy as np
import time
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import Response
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

from settings import MODEL_NAME, TRAIN_FEAT_ORDER
from utils import load_model_from_registry, feature_engineering_for_inference
from schemas import PredictionRequest, PredictionResponse, BatchPredictionRespinse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus Metrics
REQUEST_COUNT = Counter('api_request_count', 'Number of API requests', ['method', 'endpoint'])
REQUEST_LATENCY = Histogram('api_request_latency_seconds', 'Request latency seconds', ['endpoint'])
PREDICTION_COUNT = Counter('model_prediction_count', 'Count of predictions', ['result'])

app = FastAPI(
    title="ESG Score Prediction API",
    description="API for predicting ESG scores using a pre-trained ML model"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
train_feature_order = None

@app.on_event("startup")
def load_model():
    global model, train_feature_order
    logger.info("Loading model from MLflow registry...")
    train_feature_order = TRAIN_FEAT_ORDER
    model = load_model_from_registry(MODEL_NAME)
    logger.info("Model successfully loaded.")

@app.get("/")
def health():
    return {"status": "API is running"}

@app.get("/features")
def get_features():
    global train_feature_order
    if train_feature_order is None:
        return {"features": []}
    return {"features": train_feature_order}
  
@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest):
    REQUEST_COUNT.labels(method="POST", endpoint="/predict").inc()
    start_time = time.time()

    try:
        df = pd.DataFrame([payload.dict()])
        df_fe = feature_engineering_for_inference(df)
        df_fe = df_fe.reindex(columns=train_feature_order, fill_value=0)
        
        preds = model.predict(df_fe)
        pred_value = float(preds[0])

        PREDICTION_COUNT.labels(result=str(pred_value)).inc()
        REQUEST_LATENCY.labels(endpoint="/predict").observe(time.time() - start_time)

        return PredictionResponse(ESG_Score=pred_value)

    except Exception as e:
        logger.error(f"Prediction Failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch_predict", response_model=BatchPredictionRespinse)
async def batch_predict(file: UploadFile = File(...)):
    REQUEST_COUNT.labels(method="POST", endpoint="/batch_predict").inc()
    start_time = time.time()

    if file.content_type not in ("text/csv", "application/vnd.ms-excel"):
        raise HTTPException(status_code=400, detail="Invalid file type. Only CSV files are accepted")

    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        df_fe = feature_engineering_for_inference(df)

        preds = [float(x) for x in model.predict(df_fe)]
        REQUEST_LATENCY.labels(endpoint="/batch_predict").observe(time.time() - start_time)

        return BatchPredictionRespinse(ESG_Scores=preds, total=len(preds))

    except Exception as e:
        logger.error(f"Batch Prediction Failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
