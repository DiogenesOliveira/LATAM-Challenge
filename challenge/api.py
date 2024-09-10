import fastapi
from pydantic import BaseModel, validator
import pandas as pd
from typing import List, Dict, Any
from challenge.model import DelayModel

OPERAS = [
    "Aerolineas Argentinas",
    "Aeromexico",
    "Air Canada",
    "Air France",
    "Alitalia",
    "American Airlines",
    "Austral",
    "Avianca",
    "British Airways",
    "Copa Air",
    "Delta Air",
    "Gol Trans",
    "Grupo LATAM",
    "Iberia",
    "JetSmart SPA",
    "K.L.M.",
    "Lacsa",
    "Latin American Wings",
    "Oceanair Linhas Aereas",
    "Plus Ultra Lineas Aereas",
    "Qantas Airways",
    "Sky Airline",
    "United Airlines",
]

TIPOVUELOS = [
    "N",
    "I"
]

MESES = [i for i in range(1,13)]



# Initialize the FastAPI app
app = fastapi.FastAPI()

# Initialize the model
model = DelayModel()

# Health check endpoint
@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {"status": "OK"}

# Prediction input model
class FlightData(BaseModel):
    OPERA: str
    TIPOVUELO: str
    MES: int

    @validator("OPERA")
    def validate_opera(cls, value):
        if value not in OPERAS:
            raise fastapi.HTTPException(
                status_code=400
            )
        return value
    
    @validator("TIPOVUELO")
    def validate_tipovuelo(cls, value):
        if value not in TIPOVUELOS:
            raise fastapi.HTTPException(
                status_code=400
            )
        return value
    
    @validator("MES")
    def validate_mes(cls, value):
        if value not in MESES:
            raise fastapi.HTTPException(
                status_code=400
            )
        return value

class PredictionRequest(BaseModel):
    flights: List[FlightData]

# Prediction endpoint
@app.post("/predict", status_code=200)
async def post_predict(request: PredictionRequest) -> Dict[str, Any]:
    # Convert input data to DataFrame
    data = pd.DataFrame([flight.dict() for flight in request.flights])
    
    # Preprocess the input data
    features = model.preprocess(data)
    
    # Predict using the model
    predictions = model.predict(features)
    
    return {"predict": predictions}
