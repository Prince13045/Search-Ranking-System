from fastapi import FastAPI ,HTTPException
from pydantic import BaseModel

from src.Pipeline.prediction_pipeline import PredictionPipeline

app = FastAPI(
    title="Search Ranking System",
    description="Amazon-style Search Ranking API",
    version="1.0"
)

pipeline = PredictionPipeline()


class SearchRequest(BaseModel):
    query: str


@app.get("/")
def home():

    return {
        "message": "Search Ranking System API is Running"
    }


@app.post("/predict")
def predict(request: SearchRequest):

    try:

        results = pipeline.predict(request.query)

        return results.to_dict(orient="records")

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@app.get("/health")
def health():

    return {
        "status": "healthy"
    }