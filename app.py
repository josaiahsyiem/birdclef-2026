"""
app.py
------
FastAPI inference endpoint for BirdCLEF+ 2026.
Accepts a 60-second OGG audio file and returns
species detection probabilities.
"""

import os
import json
import tempfile
import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
from pathlib import Path

app = FastAPI(
    title="BirdCLEF+ 2026 Species Detector",
    description="Acoustic species identification in the Pantanal — 226th/4094 (Top 6%)",
    version="1.0.0",
)

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global model state
MODELS = None
PERCH_SESSION = None
PERCH_INPUT_NAME = None
PERCH_OUTPUT_MAP = None
PRIMARY_LABELS = None
WEIGHTS_DIR = Path("weights")


def load_primary_labels():
    """Load species labels from sample_submission.csv or fallback."""
    sample_sub_path = WEIGHTS_DIR / "sample_submission.csv"
    if sample_sub_path.exists():
        df = pd.read_csv(sample_sub_path)
        return df.columns[1:].tolist()
    return [f"species_{i}" for i in range(234)]


@app.on_event("startup")
async def startup_event():
    """Load all models at startup."""
    global MODELS, PERCH_SESSION, PERCH_INPUT_NAME, PERCH_OUTPUT_MAP, PRIMARY_LABELS

    print("Loading models...")

    try:
        from src.inference import load_models, load_perch_session
        PRIMARY_LABELS = load_primary_labels()

        # Load Perch ONNX
        perch_path = WEIGHTS_DIR / "perch_v2_no_dft.onnx"
        if perch_path.exists():
            PERCH_SESSION, PERCH_INPUT_NAME, PERCH_OUTPUT_MAP = load_perch_session(
                str(perch_path)
            )
            print(f"Perch loaded from {perch_path}")
        else:
            print(f"WARNING: Perch not found at {perch_path}")

        # Load ProtoSSM + ResidualSSM
        if WEIGHTS_DIR.exists():
            MODELS = load_models(str(WEIGHTS_DIR))
            print("ProtoSSM and ResidualSSM loaded")
        else:
            print(f"WARNING: weights directory not found at {WEIGHTS_DIR}")

    except Exception as e:
        print(f"Model loading error: {e}")


@app.get("/", response_class=HTMLResponse)
def root():
    html_path = Path("static/index.html")
    return HTMLResponse(content=html_path.read_text())


@app.get("/health")
def health():
    return {
        "status": "ok",
        "models_loaded": MODELS is not None,
        "perch_loaded": PERCH_SESSION is not None,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accept an audio file and return top species predictions.
    """
    if MODELS is None or PERCH_SESSION is None:
        return JSONResponse({
            "filename": file.filename,
            "message": "Models not loaded yet. Please try again in a moment.",
            "top_species": [],
        })

    # Save uploaded file to temp location
    contents = await file.read()
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=Path(file.filename).suffix
    ) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        from src.inference import predict as run_predict
        results = run_predict(
            audio_path=tmp_path,
            perch_session=PERCH_SESSION,
            perch_input_name=PERCH_INPUT_NAME,
            perch_output_map=PERCH_OUTPUT_MAP,
            models=MODELS,
            primary_labels=PRIMARY_LABELS,
            top_k=10,
        )
        return JSONResponse({
            "filename": file.filename,
            "top_species": results,
            "pipeline": "Perch v2 -> ProtoSSM -> ResidualSSM",
        })
    except Exception as e:
        return JSONResponse({
            "filename": file.filename,
            "error": str(e),
            "top_species": [],
        })
    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
