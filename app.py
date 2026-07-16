"""
app.py
------
FastAPI inference endpoint for BirdCLEF+ 2026.
Accepts a 60-second OGG audio file and returns
species detection probabilities.
"""

import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import uvicorn
import io

app = FastAPI(
    title="BirdCLEF+ 2026 Species Detector",
    description="Acoustic species identification in the Pantanal — 226th/4094 (Top 6%)",
    version="1.0.0",
)


@app.get("/")
def root():
    return {
        "project": "BirdCLEF+ 2026",
        "rank": "226 / 4094",
        "score": 0.949,
        "github": "https://github.com/josaiahsyiem/birdclef-2026",
        "kaggle": "https://www.kaggle.com/joesyiem",
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accept an OGG audio file and return top species predictions.
    """
    contents = await file.read()
    # Placeholder response until model weights are attached
    return JSONResponse({
        "filename": file.filename,
        "message": "Model weights not attached. Train on Kaggle and upload weights to enable full inference.",
        "pipeline": "Perch v2 -> ProtoSSM -> ResidualSSM -> TAX_SMOOTHING",
        "top_species": [],
    })


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
