from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app_model import predict_sales, ingest_data, retrain_model

app = FastAPI()

# Definición del modelo de entrada para los datos de marketing
class MarketingData(BaseModel):
    tv: float
    radio: float
    newspaper: float

# Endpoint de predicción
@app.post("/predict")
def predict(data: MarketingData):
    try:
        predicted_sales = predict_sales(data.tv, data.radio, data.newspaper)
        return {"predicted_sales": predicted_sales}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error in prediction: {str(e)}")

# Endpoint de ingesta de datos
@app.post("/ingest")
def ingest(data: MarketingData):
    try:
        new_record = ingest_data(data.tv, data.radio, data.newspaper)
        return {"message": "Data ingested successfully", "record_id": new_record.id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error in data ingestion: {str(e)}")

# Endpoint de reentrenamiento del modelo
@app.post("/retrain")
def retrain():
    
    try:
        result = retrain_model()
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error in retraining the model: {str(e)}")
