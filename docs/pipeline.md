# Pipeline Documentation

## Overview

The BirdCLEF+ 2026 solution processes 60-second passive acoustic monitoring
recordings from the Pantanal, South America. Each recording is segmented into
12 non-overlapping 5-second windows, and the pipeline outputs a probability
for each of 234 wildlife species for every window.

---

## Step 1: Audio Preprocessing

**Input:** Raw `.ogg` soundscape files (60 seconds, 32kHz)

**Process:**
- Each file is split into 12 x 5-second windows
- Windows are loaded as raw float32 waveforms
- Filename is parsed to extract recording site and UTC hour

**Output:** 12 waveform arrays of shape `(160000,)` per file

---

## Step 2: Perch v2 Feature Extraction

**Input:** Raw waveform windows

**Process:**
- Google's Perch v2 model (pretrained on 10,000+ bird species) runs on each 5-second window
- Perch outputs two things per window:
  - `embeddings`: 1536-dimensional feature vector
  - `logits`: raw scores for species it was trained on

**Why Perch?**
Perch was trained by Google at scale on a massive bird vocalization dataset.
Using it as a frozen backbone means we benefit from that pretraining without
needing to train a large model ourselves. This is the single biggest reason
Model_74 dominates the ensemble.

**Output:** Per-window embeddings `(n_windows, 1536)` and logits `(n_windows, n_perch_classes)`

**Note:** Perch results are cached to disk after the first run. Subsequent
training runs load from cache, saving 30-60 minutes per run.

---

## Step 3: Prior Table Construction

**Input:** Training soundscape labels + recording metadata (site, hour)

**Process:**
Three levels of prior probability tables are computed from training data:

- **Global prior:** species base rate across all recordings
- **Site prior:** species rate per recording site (shrinkage-weighted)
- **Hour prior:** species rate per UTC hour with circular Gaussian smoothing

The circular Gaussian smoothing (sigma=1.5 hours) on the hour prior handles
the fact that species activity peaks (dawn chorus, dusk chorus) are continuous
across hour boundaries rather than discrete jumps.

**Output:** Prior tables used to calibrate Perch logits

---

## Step 4: MLP Probes

**Input:** Perch embeddings `(n_windows, 1536)`

**Process:**
- Embeddings are standardized and PCA-reduced to 128 dimensions
- One MLPClassifier (2 hidden layers: 256 -> 128) is trained per species class
- Classes with fewer than 5 positive examples are skipped
- Per-class probes are later vectorized into a single PyTorch module for fast batch inference

**Why per-class probes?**
Each species has a different decision boundary in embedding space. Training
a separate classifier per species (rather than one multi-class model) allows
each probe to specialize on its own positive/negative distribution.

**Output:** Per-class probability scores `(n_windows, n_classes)`

---

## Step 5: ProtoSSM v2

**Input:** Perch embeddings + raw logits `(n_windows, 1536)` and `(n_windows, n_classes)`

**Process:**
ProtoSSM is a temporal sequence model that captures dependencies across the
12 windows of a recording:

1. **Input projection:** 1536 -> d_model (320 for v5, 256 for v2)
2. **Metadata embedding:** site ID and UTC hour are embedded and added
3. **SSM layers:** 4 State Space Model layers model temporal dynamics
4. **Cross-attention:** each window attends to all other windows in the file
5. **Prototype similarity:** species logits computed as cosine similarity to learned prototype vectors
6. **Fusion:** learnable blend of prototype similarity and direct head output

**Why SSM instead of Transformer?**
State Space Models have linear complexity in sequence length, making them
efficient for the 12-window sequences. They also handle irregular temporal
patterns well, which matters for species with variable call rates.

**Output:** Per-window species logits `(n_windows, n_classes)`

---

## Step 6: ResidualSSM

**Input:** ProtoSSM output logits `(n_windows, n_classes)`

**Process:**
A lightweight 2-layer SSM learns a small additive correction to the
ProtoSSM predictions. The correction is scaled by correction_weight=0.35
to prevent over-fitting.

**Why residual correction?**
ProtoSSM can make systematic errors for certain species or recording
conditions. The ResidualSSM learns to fix these without needing to retrain
the full ProtoSSM.

**Output:** Corrected logits `(n_windows, n_classes)`

---

## Step 7: xSED Rank Blend (inside Model_74)

**Input:**
- ProtoSSM probabilities `(n_windows, n_classes)`
- Distilled SED probabilities `(n_windows, n_classes)`

**Process:**
Both prediction arrays are converted to percentile ranks, then blended
with weights 0.60 for ProtoSSM and 0.40 for SED.

Rank-based blending is used instead of probability blending because the
two models have different probability scales and calibrations. Converting
to percentile ranks normalizes both to the same scale before combining.

**Output:** Rank-blended predictions `(n_windows, n_classes)`

---

## Step 8: 5-Gate Post-Processing

Applied sequentially to the rank-blended predictions:

| Gate | Name | Action |
|------|------|--------|
| 1 | Noise suppression | Reduce signal where ProtoSSM is confident but SED strongly disagrees |
| 2 | Temporal continuity | Protect continuous calls that span multiple windows |
| 3 | SED spike preservation | Keep brief sharp detections that ProtoSSM missed |
| 4 | Sonotype mirroring | Max-pool predictions across acoustically identical species groups |
| 5 | Rare class thresholding | Suppress weak Amphibia, Mammalia, and Reptilia detections |

---

## Step 9: Division Attention Ensemble Blend

**Input:** Submission CSVs from all three models

**Process:**
The three model outputs are combined using normalized weights:

- Model_21 (Distilled SED + ProtoSSM): weight 0.014
- Model_52 (ProtoSSM sub-output): weight 0.021
- Model_74 (Karnakbayev pipeline): weight 0.965

Weights are normalized to sum to 1. The extreme skew toward Model_74
reflects that it is objectively the strongest individual model. Models
21 and 52 contribute small amounts of diversity that marginally improve
the ensemble score.

**Output:** Blended submission `(n_windows, n_classes)`

---

## Step 10: TAX_SMOOTHING

**Input:** Blended submission

**Process:**
Two-level taxonomy-aware smoothing is applied after blending:

**Genus level (alpha=0.15):** For each genus with multiple species in the
dataset, each species probability is pulled slightly toward the genus mean.
Rationale: if one Amazona species is detected, related Amazona species are
more likely to also be present.

**Class level (alpha=0.05):** Lighter smoothing across the broader
taxonomic class (Aves, Amphibia, Insecta, Mammalia, Reptilia).

Smoothing formula: smoothed = (1 - alpha) x original + alpha x group_mean

**Why this works:**
Species within the same genus share habitat, diet, and behavioral patterns.
Their co-occurrence probability is higher than random. The smoothing exploits
this ecological signal without requiring explicit co-occurrence modeling.

**Output:** Final `submission.csv`

---

## Leaderboard Score Breakdown

| Component | LB Score | Ensemble Weight |
|-----------|----------|-----------------|
| Model_21 (Distilled SED + ProtoSSM) | 0.928 | 1.4% |
| Model_52 (ProtoSSM sub-output) | 0.949 | 2.1% |
| Model_74 (Karnakbayev pipeline) | 0.949 | 96.5% |
| **Final ensemble + TAX_SMOOTHING** | **0.949** | — |

**Final rank: 226 / 4,094 (Top 6%)**
