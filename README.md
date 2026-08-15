# BirdCLEF+ 2026 — Acoustic Species Identification in the Pantanal

> **226th place out of 4,094 teams (Top 6%)** | Public Leaderboard Score: **0.949**
> Kaggle Competition: [BirdCLEF+ 2026](https://www.kaggle.com/competitions/birdclef-2026)
> Kaggle: [joesyiem](https://www.kaggle.com/joesyiem)
> **Live Demo:** [birdclef-detector on Azure](https://birdclef-detector-e8c9fthzc4a5c4da.centralindia-01.azurewebsites.net)
> **Video Walkthrough:** [Watch on YouTube](https://youtu.be/nFpv6Mb8TXQ)

---

## Overview

BirdCLEF+ 2026 is a bioacoustics machine learning competition hosted on Kaggle. The objective is to identify **234 wildlife species** across **five taxonomic classes** (Aves, Amphibia, Insecta, Mammalia, Reptilia) from passive acoustic recordings collected in the **Pantanal** region of South America. Each 60-second recording is divided into twelve 5-second segments, and the model predicts the probability of every species for each segment.

This repository contains the complete competition solution — model architectures, training pipeline, post-processing methods, and ensemble strategy — refactored from the original competition notebook into a modular package and deployed as a containerized inference service.

---

## Result

| Metric | Value |
| --- | --- |
| Public Leaderboard Score | **0.949** |
| Final Rank | **226 / 4,094** |
| Percentile | **Top 6%** |
| Evaluation Metric | Macro-averaged ROC-AUC |
| Species | 234 across 5 taxonomic classes |
| Dataset Size | 16.14 GB (46,200+ files) |

---

## Solution Architecture

The final submission is a **three-model ensemble** with taxonomy-aware post-processing. Each model operates on 5-second audio segments extracted from the original 60-second recordings.

```text
Raw Audio (60s OGG)
        |
        v
  5s Window Segmentation (12 windows per file)
        |
        +--------------------------------------------------+
        |                                                  |
        v                                                  v
  Perch v2 Backbone                          Mel Spectrogram
  (Google, frozen)                           (256 mels, 32kHz)
        |                                                  |
        +-- 1536-dim embeddings                            |
        +-- 234-class logits                               v
        |                                       EfficientNet-B0 SED
        +--------------+---------------+         (Distilled)
        |              |               |
        v              v               v
  Perch Embedding  Sequence Model   Distilled Spectrogram
     Pipeline         Variant             Detector
      (0.965)         (0.021)              (0.014)
        |              |               |
        +--------------+---------------+
                       |
                       v
              Rank-normalised Ensemble
                       |
                       v
        Taxonomy-aware Post-processing
                       |
                       v
                Final submission.csv
```

---

## Models

### Perch Embedding Pipeline — primary model

Contributes the majority of the final ensemble score.

**Architecture**

* Frozen **Perch v2** backbone extracting 1536-dimensional audio embeddings
* PCA reduction for feature compression
* Per-class MLP probes (256 to 128 hidden layers for frequent classes, 128 to 64 for rare)
* ProtoSSM with ResidualSSM correction for temporal refinement
* Shrinkage-weighted site and hour prior tables with circular Gaussian smoothing (sigma = 1.5h)

**Post-processing pipeline**

1. Noise suppression for inconsistent predictions
2. Temporal smoothing using a fat-tailed t-distribution kernel with a 35-second context window
3. Preservation of strong spectrogram-detector detections
4. Sonotype mirroring for acoustically identical species
5. Adaptive thresholding for rare amphibian, mammal, and reptile classes

Final prediction combines the sequence model and spectrogram detector outputs using a rank-normalised weighted blend of **0.60 / 0.40**.

**Ensemble weight:** 0.965 | **Standalone LB:** 0.949

---

### Sequence Model Variant

An intermediate checkpoint from the sequence-model training pipeline, captured before the final blend and retained for ensemble diversity.

**Ensemble weight:** 0.021 | **Standalone LB:** 0.949

---

### Distilled Spectrogram Detector

Combines a distilled EfficientNet-B0 sound event detector with a temporal sequence model.

**Architecture**

* EfficientNet-B0 (`tf_efficientnet_b0.ns_jft_in1k`) trained on 256-bin mel spectrograms at 32 kHz
* Knowledge distillation from frozen **Perch v2** embeddings (1536 dimensions) using MSE loss
* EfficientNet encoder followed by a state-space temporal model with cross-attention

**Training**

* 25 epochs, 5-fold Stratified K-Fold cross-validation
* Focal loss with label smoothing
* Stochastic Weight Averaging (SWA)
* MixUp and SpecAugment augmentation

**Ensemble weight:** 0.014 | **Standalone LB:** 0.928

---

## Post-processing

### Taxonomy-aware smoothing

After combining predictions from the three models, taxonomy-aware smoothing improves consistency across related species.

* **Genus-level smoothing (alpha = 0.15)** — species in the same genus have prediction scores adjusted toward the genus mean, since closely related species share vocal characteristics and habitat
* **Class-level smoothing (alpha = 0.05)** — lighter smoothing across broader taxonomic groups

Formula: `smoothed = (1 - alpha) * original + alpha * group_mean`

This approach was identified by analysing high-performing public solutions and consistently improved prediction stability.

### Ensemble blending

Outputs from the three models are converted to percentile ranks — normalising for differing probability calibrations — and combined via weighted blending. The Perch embedding pipeline receives most of the weight given its strongest individual performance.

---

## Repository Structure

```text
birdclef-2026/
├── configs/
│   └── ensemble_config.yaml     # Weights, paths, hyperparameters
├── docs/
│   └── pipeline.md              # Detailed pipeline documentation
├── scripts/
│   ├── train_model21.py         # Train spectrogram detector
│   ├── train_model74.py         # Train Perch embedding pipeline
│   └── predict.py               # Full ensemble inference
├── src/
│   ├── models/                  # Model architectures
│   ├── postprocessing/          # Taxonomy smoothing + temporal gates
│   ├── ensemble.py              # Blending logic
│   ├── inference.py             # Deployment inference pipeline
│   └── utils.py                 # Shared utilities
├── static/
│   └── index.html               # Web frontend
├── app.py                       # FastAPI server
├── Dockerfile                   # Container definition
├── .dockerignore
├── requirements.txt
└── README.md
```

---

## Deployment

The inference service is containerized and deployed on **Microsoft Azure App Service**.

| Component | Details |
| --- | --- |
| **Container Registry** | Docker Hub — `josaiahsyiem/birdclef-2026:latest` |
| **Hosting** | Azure App Service (Linux, Container mode, Central India) |
| **Model Weights** | [Hugging Face Hub](https://huggingface.co/josaiahsyiem/birdclef-2026-weights) |
| **API Framework** | FastAPI with gunicorn and a uvicorn ASGI worker |
| **Base Image** | `python:3.10-slim` with libsndfile1 and ffmpeg |

Model weights are hosted on Hugging Face Hub and downloaded at container startup via `huggingface_hub.hf_hub_download`:

* `perch_v2_no_dft.onnx` (413 MB) — Perch v2 backbone in ONNX format
* `proto_ssm_74.pt` (2.9 MB) — sequence model weights
* `residual_ssm_best.pt` (1.8 MB) — residual correction weights
* `site2i_74.json` — site-to-index mapping
* `taxonomy.csv` — species taxonomy for common and scientific name lookup
* `sample_submission.csv` — defines the 234 output columns

### Inference API

| Endpoint | Method | Description |
| --- | --- | --- |
| `/` | GET | Web interface |
| `/health` | GET | Model load status |
| `/docs` | GET | Interactive OpenAPI documentation |
| `/predict` | POST | Audio upload, returns top species with confidences |

### Inference pipeline

```text
audio upload -> librosa load (32kHz mono) -> 12 x 5s windows
-> Perch v2 ONNX -> 1536-dim embeddings
-> sequence model -> residual correction (weight 0.35)
-> sigmoid -> max across windows -> top-k species
-> taxonomy lookup for common and scientific names
```

### Performance note

The service runs on **Basic B1 (1.75 GB RAM, shared vCPU)**. Inference is CPU-only, so a 60-second recording takes noticeably longer than on a GPU. Three choices keep the deployment within this budget:

* **CPU-only PyTorch** via `--extra-index-url https://download.pytorch.org/whl/cpu`, which cuts image size substantially and reduces container pull time
* **A single gunicorn worker**, since each worker loads its own copy of the models
* **Always On enabled**, so the container stays resident rather than re-pulling the image after idle periods

---

## Reproducing the Solution

### Requirements

* Python 3.10 or later
* CUDA-enabled GPU (8 GB VRAM or higher recommended for training)
* BirdCLEF+ 2026 competition dataset
* Perch v2 model weights

### Installation

```bash
git clone https://github.com/josaiahsyiem/birdclef-2026.git
cd birdclef-2026
pip install -r requirements.txt
```

Update `configs/ensemble_config.yaml` with your dataset and checkpoint paths before training or inference.

### Running locally

```bash
python app.py          # http://localhost:7860
```

### Running via Docker

```bash
docker build -t birdclef-2026 .
docker run -p 8000:8000 birdclef-2026
```

---

## Lessons Learned

1. **Strong baselines matter.** High-performing public notebooks are usually carefully tuned; changing parameters without clear justification consistently reduced performance (0.949 to 0.945).

2. **Not every idea improves the model.** Generating pseudo-labels from Perch embeddings to retrain another Perch-based model introduces no new signal — the training loop becomes circular.

3. **Understanding successful solutions is valuable.** Analysing techniques common to high-ranking public notebooks identified taxonomy-aware post-processing as the highest-impact change.

4. **Perch v2 is a powerful feature extractor.** Frozen Perch embeddings with lightweight task-specific heads outperformed training a CNN from scratch, reflecting the value of large-scale pretraining across 10,000+ species.

5. **Deployment constraints shape design.** Fitting the pipeline into a 1.75 GB container required CPU-only dependencies, single-worker serving, and moving model weights out of the image — decisions that never surface during local development.

---

## Dependencies

Core libraries:

* PyTorch — sequence models and spectrogram detector
* ONNX Runtime — Perch v2 inference
* timm — EfficientNet-B0 backbone
* librosa, torchaudio, soundfile — audio processing
* scikit-learn — MLP probes, PCA, cross-validation, isotonic calibration
* NumPy, SciPy, pandas — numerical and tabular processing
* FastAPI, uvicorn, gunicorn — inference service
* huggingface-hub — model weight distribution

See `requirements.txt` for the complete list.

---

## Acknowledgements

This solution builds on ideas and open-source work shared by the BirdCLEF community:

* Tucker Arrants
* hideyukizushi (yukiZ)
* Yaroslav Kholmirzayev
* F.A. Nina

---

## License

Released under the MIT License. See the `LICENSE` file for details.
