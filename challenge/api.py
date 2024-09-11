from typing import Any, Dict, List

import fastapi
import pandas as pd
from pydantic import BaseModel, validator

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
TIPOVUELOS = ["I", "N"]
MESES = [i for i in range(1,13)]


app = fastapi.FastAPI()
model = DelayModel()

@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {"status": "OK"}

class FlightData(BaseModel):
    OPERA: str
    TIPOVUELO: str
    MES: int

    @validator("OPERA")
    def validate_opera(cls, value):
        if value not in OPERAS:
            raise fastapi.HTTPException(
                status_code=400,
                detail="Please, enter a valid OPERA."
            )
        return value
    
    @validator("TIPOVUELO")
    def validate_tipovuelo(cls, value):
        if value not in TIPOVUELOS:
            raise fastapi.HTTPException(
                status_code=400,
                detail="TIPOVUELO must be I or N."
            )
        return value
    
    @validator("MES")
    def validate_mes(cls, value):
        if value not in MESES:
            raise fastapi.HTTPException(
                status_code=400,
                detail="MES must be an integer between 1 and 12."
            )
        return value

class PredictionRequest(BaseModel):
    flights: List[FlightData]

@app.post("/predict", status_code=200)
async def post_predict(request: PredictionRequest) -> Dict[str, Any]:
    data = pd.DataFrame([flight.dict() for flight in request.flights])
    features = model.preprocess(data)
    predictions = model.predict(features)
    return {"predict": predictions}
