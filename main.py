from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from src.ml.pipelines import (
    test_full_preprocessing,
    train_full_preprocessing,
    training, predicting,
)
from src.tools.data_config import (
    BKI_LOCAL_PATH,
    TARGET_COLUMN,
    TARGET_LOCAL_PATH,
    TEST_LOCAL_PATH,
)
from src.tools.logger import logger

app = FastAPI()


class TrainRequest(BaseModel):
    features_data_path: str = BKI_LOCAL_PATH
    target_data_path: str = TARGET_LOCAL_PATH


class PredictRequest(BaseModel):
    test_data_path: str = TEST_LOCAL_PATH


@app.post("/train")
def train_model(request: TrainRequest):

    if not os.path.exists(request.features_data_path):
        raise HTTPException(
            status_code=404,
            detail="Features data file not found",
        )
    if not os.path.exists(request.target_data_path):
        raise HTTPException(
            status_code=404,
            detail="Target data file not found",
        )
    logger.info('Start training pipeline')
    training(train_full_preprocessing())

    return {"message": "Model trained and saved successfully"}


@app.post("/predict")
def predict(request: PredictRequest):
    if not os.path.exists(request.test_data_path):
        raise HTTPException(
            status_code=404,
            detail="Test data file not found",
        )

    logger.info('Start predicting pipeline')
    predicting(test_full_preprocessing())
    return {"message": "Predictions saved successfully"}


@app.get("/")
def read_root():
    return {
        "message": "Welcome to the ML API. Use /train to train the model and /predict to make predictions."
    }
