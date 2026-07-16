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
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import io
from pathlib import Path

app = FastAPI(
    title="BirdCLEF+ 2026 Species Detector",
    description="Acoustic species identification in the Pantanal — 226th/4094 (Top 6%)",
    version="1.0.0",
)

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
def root():
    html_path = Path("static/index.html")
    return HTMLResponse(content=html_path.read_text())


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accept an OGG audio file and return top species predictions.
    """
    contents = await file.read()
    return JSONResponse({
        "filename": file.filename,
        "message": "Model weights not attached yet. Train on Kaggle and upload weights to enable full inference.",
        "pipeline": "Perch v2 -> ProtoSSM -> ResidualSSM -> TAX_SMOOTHING",
        "top_species": [],
    })


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
