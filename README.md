# BirdCLEF+ 2026: Acoustic Species Identification in the Pantanal

> **226th place out of 4,094 teams (Top 6%)** | Public Leaderboard Score: **0.949**  
> Kaggle Competition: [BirdCLEF+ 2026](https://www.kaggle.com/competitions/birdclef-2026)  
> Kaggle: (https://www.kaggle.com/joesyiem)

---

## Overview

BirdCLEF+ 2026 is a bioacoustics machine learning competition hosted on Kaggle. The objective is to identify **234 wildlife species** (birds, amphibians, insects, mammals, and reptiles) from passive acoustic recordings collected in the **Pantanal** region of South America. Each 60-second audio recording is divided into twelve 5-second segments, and the model predicts the probability of every species for each segment.

This repository contains my complete competition solution, including the model architectures, training pipeline, post-processing methods, and ensemble strategy. The original competition notebook has been cleaned, modularized, and documented to make the project easier to understand, reproduce, and extend.

---

## Result

| Metric                   | Value                  |
| ------------------------ | ---------------------- |
| Public Leaderboard Score | **0.949**              |
| Final Rank               | **226 / 4,094**        |
| Percentile               | **Top 6%**             |
| Evaluation Metric        | Macro-averaged ROC-AUC |

---

## Solution Architecture

The final submission is a **three-model ensemble** with taxonomy-aware post-processing. Each model operates on 5-second audio segments extracted from the original 60-second recordings.

```text
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
        ├──────────────┬───────────────┐         (Distilled)
        │              │               │
        ▼              ▼               ▼
   Model_21        Model_52        Model_74
  (ProtoSSM)     (Intermediate)   (Main Model)
        │              │               │
        ▼              ▼               ▼
   subm_21.csv   subm_52p.csv    subm_74.csv
        │              │               │
        └──────────────┴───────────────┘
                       │
                       ▼
                 Weighted Ensemble
             [0.014, 0.021, 0.965]
                       │
                       ▼
        Taxonomy-aware Post-processing
                       │
                       ▼
                Final submission.csv
```

---

## Models

### Model_21

This model combines a distilled EfficientNet-B0 sound event detector with a ProtoSSM temporal model.

**Architecture**

* EfficientNet-B0 (`tf_efficientnet_b0.ns_jft_in1k`) trained on 256-bin mel spectrograms sampled at 32 kHz
* Knowledge distillation from frozen **Perch v2** embeddings (1536 dimensions) using MSE loss
* EfficientNet encoder followed by **ProtoSSM v5**, which models temporal relationships using four state-space layers with cross-attention (`d_model=320`, `d_state=32`)

**Training**

* 25 training epochs
* 5-fold Stratified K-Fold cross-validation
* Focal loss with label smoothing
* Stochastic Weight Averaging (SWA)
* MixUp augmentation
* SpecAugment

**Ensemble weight:** **0.014**

---

### Model_52

Model_52 is an intermediate output from the ProtoSSM training pipeline used to increase ensemble diversity. Its predictions are saved separately as `subm_52p.csv`.

**Ensemble weight:** **0.021**

---

### Model_74

Model_74 is the primary model and contributes the majority of the final ensemble score.

**Architecture**

* Frozen **Perch v2** backbone for extracting 1536-dimensional audio embeddings
* PCA for feature reduction
* Class-specific MLP probes
* ProtoSSM v2 with ResidualSSM for temporal refinement

**Post-processing pipeline**

* Noise suppression for inconsistent predictions
* Temporal smoothing using a fat-tailed t-distribution kernel with a 35-second context window
* Preservation of strong SED detections
* Sonotype mirroring for acoustically similar species
* Adaptive thresholding for rare amphibian, mammal, and reptile classes

The final prediction from this model is obtained by combining ProtoSSM and distilled SED outputs using a weighted blend of **0.60** and **0.40**.

**Ensemble weight:** **0.965**


---

## Post-processing

### Taxonomy-aware smoothing

After combining the predictions from the three models, a taxonomy-aware smoothing step is applied to improve consistency across related species.

* **Genus-level smoothing (`α = 0.15`)**
  Species belonging to the same genus have their prediction scores adjusted slightly toward the genus average. Since closely related species often share similar vocal characteristics, this helps reduce noisy predictions.

* **Class-level smoothing (`α = 0.05`)**
  A lighter smoothing step is applied across broader taxonomic groups such as birds, amphibians, mammals, and reptiles.

This approach was inspired by several high-performing public BirdCLEF solutions and consistently improved the stability of the final predictions.

### Ensemble blending

The outputs from the three models are combined using weighted blending. Since Model_74 produced the strongest individual performance, it receives most of the ensemble weight (0.965), while Model_21 and Model_52 provide complementary predictions.

---

## Repository Structure

```text
birdclef-2026/
├── configs/
├── docs/
├── notebooks/
├── scripts/
├── src/
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

The repository is organized into separate modules for training, inference, post-processing, model definitions, and documentation, making it easier to understand and extend than the original competition notebook.

---

## Reproducing the Solution

### Requirements

* Python 3.10 or later
* CUDA-enabled GPU (8 GB VRAM or higher recommended)
* BirdCLEF+ 2026 competition dataset
* Perch v2 model weights

### Installation

```bash
git clone https://github.com/josaiahsyiem/birdclef-2026.git
cd birdclef-2026
pip install -r requirements.txt
```

Update the configuration file with the locations of your datasets and model checkpoints before training or inference.

Training and inference scripts are provided in the `scripts/` directory.

---

## Lessons Learned

Some of the biggest takeaways from this competition were:

1. **Strong baselines matter.** Public notebooks that perform well are usually carefully tuned, and changing parameters without a clear reason often reduced performance.

2. **Not every idea improves the model.** Some experiments, such as generating pseudo-labels from Perch embeddings and retraining another Perch-based model, did not provide meaningful gains.

3. **Understanding successful solutions is valuable.** Studying techniques used by high-ranking public notebooks helped identify ideas that were worth exploring, including taxonomy-aware post-processing.

4. **Perch v2 is a powerful feature extractor.** Using frozen Perch embeddings with lightweight task-specific models proved more effective than training a CNN from scratch for this competition.

---

## Dependencies

This project primarily uses:

* PyTorch
* TensorFlow
* ONNX Runtime
* timm
* torchaudio
* librosa
* scikit-learn
* NumPy
* SciPy
* pandas

See `requirements.txt` for the complete list of dependencies.

---

## Acknowledgements

This solution builds upon ideas and open-source work shared by the BirdCLEF community. In particular, I would like to acknowledge:

* Tucker Arrants
* hideyukizushi (yukiZ)
* Yaroslav Kholmirzayev
* F.A. Nina

Their public notebooks provided valuable insights and served as important references during the competition.

---

## License

This project is released under the MIT License. See the `LICENSE` file for more information.

