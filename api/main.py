from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel
import yaml
import joblib
import pandas as pd
from src.features import add_engineered_features

app = FastAPI()

with open("config.yaml") as f:
    config = yaml.safe_load(f)

model = joblib.load(config["model"]["path"])

# Pydantic model to ensure inputs follow specific valid data types
class MachineInput(BaseModel):
    type: Literal["L", "M", "H"]
    air_temp_k: float
    process_temp_k: float
    rotational_speed_rpm: float
    torque_nm: float
    tool_wear_min: float

@app.post("/predict")
def predict(input_data: MachineInput):
    data_form = input_data.model_dump()
    input_df = pd.DataFrame([data_form])

    predictive_df = add_engineered_features(input_df)

    X = predictive_df[config["features"]["numeric"] + config["features"]["categorical"]]

    proba = model.predict_proba(X)[:, 1]
    prediction = (proba >= config["evaluation"]["threshold"]).astype(int)

    proba = float(proba[0])

    prediction = int(prediction[0])

    return {"failure_probability": proba, "prediction": prediction}

@app.get("/")
def health():
    return {"status": "ok"}

