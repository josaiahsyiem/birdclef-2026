# BirdCLEF+ 2026 — Acoustic Species Identification in the Pantanal

> **226th place out of 4,094 teams (Top 6%)** | Public Leaderboard Score: **0.949**  
> Kaggle Competition: [BirdCLEF+ 2026](https://www.kaggle.com/competitions/birdclef-2026)  
> Kaggle: (https://www.kaggle.com/joesyiem)

---

## Overview

BirdCLEF+ 2026 is a bioacoustics machine learning competition hosted on Kaggle. The task is to identify **234 wildlife species** — birds, amphibians, insects, mammals, and reptiles — from passive acoustic monitoring recordings captured in the **Pantanal**, South America. Audio files are 60-second soundscapes segmented into 5-second windows, and the model must output a probability for each of the 234 species for every window.

This repository contains the full solution: model architectures, training pipelines, post-processing logic, and ensemble code — extracted, cleaned, and modularized from the original competition notebook.

---

## Result

| Metric | Value |
|---|---|
| Public Leaderboard Score | 0.949 |
| Final Rank | 226 / 4,094 |
| Percentile | Top 6% |
| Evaluation Metric | Macro-averaged ROC-AUC |

---

## Solution Architecture

The solution is a **three-model ensemble** with taxonomy-aware post-processing. Each model operates on 5-second audio windows extracted from 60-second soundscape recordings.

```
Raw Audio (60s OGG)
        │
        ▼
  5s Window Segmentation (12 windows per file)
        │
        ├──────────────────────────────────────────────────┐
        │                                                  │
        ▼                                                  ▼
  Perch v2 Backbone                          Mel Spectrogram
  (Google, frozen)                           (256 mels, 32kHz)
        │                                                  │
        ├── 1536-dim embeddings                            │
        ├── 234-class logits                               ▼
        │                                       EfficientNet-B0 SED
        ├──────────────┬───────────────┐         (Distilled, Tucker Arrants)
        │              │               │                   │
        ▼              ▼               ▼                   ▼
   Model_21        Model_52        Model_74          submission_sed.csv
  (ProtoSSM)    (ProtoSSM        (Karnakbayev
                sub-output)      Full Pipeline)
        │              │               │
        ▼              ▼               ▼
   subm_21.csv   subm_52p.csv    subm_74.csv
        │              │               │
        └──────────────┴───────────────┘
                       │
                       ▼
              Division Attention Blend
              weights: [0.014, 0.021, 0.965]
                       │
                       ▼
            TAX_SMOOTHING Post-Processing
            (genus α=0.15, class α=0.05)
                       │
                       ▼
               submission.csv
```

---

## Models

### Model_21 — Distilled EfficientNet SED + ProtoSSM
- **Backbone:** `tf_efficientnet_b0.ns_jft_in1k` trained on mel spectrograms (256 mels, 32kHz)
- **Distillation:** Knowledge distillation from frozen Perch v2 embeddings (1536-dim) using MSE loss
- **Architecture:** EfficientNet-B0 encoder → ProtoSSM v5 (Prototype State Space Model)
- **ProtoSSM:** SSM-based temporal sequence model with cross-attention, d_model=320, d_state=32, 4 SSM layers
- **Training:** 25 epochs, 5-fold StratifiedKFold, focal loss + label smoothing, SWA, MixUp, SpecAugment
- **Ensemble weight:** 0.014

### Model_52 — ProtoSSM Sub-Output
- Intermediate output of the ProtoSSM pipeline from Model_21's training block
- Saved separately as `subm_52p.csv` for ensemble diversity
- **Ensemble weight:** 0.021

### Model_74 — Karnakbayev Full Pipeline (dominant model)
- **Backbone:** Perch v2 (frozen) — Google's bird vocalization classifier pretrained on 10,000+ species
- **Embedding pipeline:** 1536-dim Perch embeddings → PCA reduction → MLP probes (per-class)
- **Temporal model:** ProtoSSM v2 + ResidualSSM for sequence-level refinement
- **Post-processing:** 5-gate pipeline:
  - Gate 1: Noise suppression (ProtoSSM confident, SED disagrees)
  - Gate 2: Temporal continuity (fat-tailed t-distribution kernel, 35s context)
  - Gate 3: SED spike preservation
  - Gate 4: Sonotype mirroring (acoustically identical species groups)
  - Gate 5: Adaptive rare-class thresholding (Amphibia, Mammalia, Reptilia)
- **xSED blend:** Rank-based blend of ProtoSSM and Distilled SED outputs (0.60 / 0.40)
- **Ensemble weight:** 0.965

---

## Post-Processing

### TAX_SMOOTHING
The key insight that separates this solution from lower-scoring baselines. After ensemble blending, a taxonomy-aware smoothing pass is applied:

- **Genus-level smoothing (α=0.15):** Species sharing the same genus have their probabilities pulled slightly toward the genus mean. Rationale: if one species of *Amazona* is detected, related *Amazona* species are more likely to be present.
- **Class-level smoothing (α=0.05):** Lighter smoothing across the broader taxonomic class (Aves, Amphibia, etc.)

This post-processing step was identified as a shared pattern across all public notebooks scoring 0.950+ on the leaderboard.

### Division Attention Blending
The three model outputs are blended using division-attention weighting rather than simple linear combination, which provides more stable ensemble behavior when one model dominates (Model_74 at 96.5% weight).

---

## Repository Structure

```
birdclef-2026/
├── configs/
│   └── ensemble_config.yaml       # Ensemble weights, paths, hyperparameters
├── docs/
│   └── pipeline.md                # Detailed pipeline walkthrough
├── notebooks/
│   └── birdclef-2026-eos-7-sz.ipynb  # Original competition notebook (frozen)
├── scripts/
│   ├── train_model21.py           # Train EfficientNet SED + ProtoSSM
│   ├── train_model74.py           # Train Karnakbayev pipeline
│   └── predict.py                 # Run full inference pipeline
├── src/
│   ├── models/
│   │   ├── distilled_sed.py       # EfficientNet SED architecture
│   │   ├── proto_ssm.py           # ProtoSSM v2/v5 architecture
│   │   └── karnakbayev.py         # Full Karnakbayev pipeline
│   ├── postprocessing/
│   │   ├── tax_smoothing.py       # Taxonomy-aware smoothing
│   │   ├── temporal_gates.py      # 5-gate post-processing pipeline
│   │   └── xsed_blend.py          # Rank-based SED blend
│   ├── ensemble.py                # Division attention blending
│   └── utils.py                   # Shared utilities
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

---

## Reproduction

### Prerequisites
- Python 3.10+
- CUDA-capable GPU (8GB+ VRAM recommended)
- BirdCLEF+ 2026 competition data (available on Kaggle)
- Perch v2 model weights (available via Kaggle model hub)

### Setup

```bash
git clone https://github.com/josaiahsyiem/birdclef-2026.git
cd birdclef-2026
pip install -r requirements.txt
```

### Configure paths

Edit `configs/ensemble_config.yaml` to point to your local data directory.

### Train Model_74 (recommended starting point)

```bash
python scripts/train_model74.py --config configs/ensemble_config.yaml
```

### Train Model_21

```bash
python scripts/train_model21.py --config configs/ensemble_config.yaml
```

### Run inference

```bash
python scripts/predict.py --config configs/ensemble_config.yaml --output submission.csv
```

---

## Key Learnings

**1. Don't blindly tune public notebook parameters.**
The EoS baseline was carefully optimized. Early experiments changing ensemble weights and blending parameters caused score drops from 0.949 to 0.945. The original parameters were restored.

**2. Pseudo-labeling pitfall.**
Using Perch embeddings to generate pseudo-labels for retraining a Perch-embedding-based model adds no new signal — it is circular by design and confirmed as a failure mode in this competition.

**3. Read the leaderboard before experimenting.**
Studying what public notebooks scoring at the target level had in common (TAX_SMOOTHING) was higher ROI than ad hoc experimentation. This single insight was the difference between 0.945 and 0.950.

**4. Perch v2 as a frozen backbone is extremely powerful.**
Model_74 dominates the ensemble at 96.5% weight precisely because Perch v2 was pretrained on 10,000+ bird species at Google scale. Fine-tuning small heads on top of frozen Perch embeddings outperforms training CNN-based models from scratch for this task.

---

## Dependencies

Key libraries used:
- `torch` — PyTorch for model training
- `tensorflow` — Perch v2 SavedModel inference
- `onnxruntime` — ONNX Perch inference (faster)
- `timm` — EfficientNet backbone
- `torchaudio` — Audio processing
- `librosa` — Mel spectrogram computation
- `scikit-learn` — MLP probes, isotonic calibration, cross-validation
- `pandas`, `numpy`, `scipy` — Data processing

See `requirements.txt` for pinned versions.

---

## Acknowledgements

This solution builds on the work of several public Kaggle contributors:
- **Tucker Arrants** — BC2026 Distilled SED notebook (Model_21 backbone)
- **yukiZ (hideyukizushi)** — Perch + ProtoSSM + ResSSM pipeline (Model_21/52)
- **Yaroslav Kholmirzayev** — v6_0949_replay pipeline (Model_74)
- **F.A.Nina** — EoS.7 ensemble framework

All original notebooks are publicly available on the BirdCLEF+ 2026 Kaggle competition page.

---

## License

MIT License. See [LICENSE](LICENSE) for details.
