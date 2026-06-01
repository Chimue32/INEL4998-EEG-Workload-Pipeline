# INEL 4998 — EEG Mental Workload Classification Pipeline
**Universidad de Puerto Rico — Mayagüez (UPRM)**
**Artificial Intelligence Imaging Group (AIIG Lab)**
**Investigator:** Favian O. Díaz | **Advisor:** Prof. Vidya Manian

---

## Overview
This repository contains the machine learning pipeline developed for INEL 4998 (Undergraduate Research) as a preliminary validation step toward building a **Neurocognitive Readiness System for Baseball Pitchers** — the High-Leverage Readiness (HLR) system described in the original research proposal.

The pipeline classifies EEG-based mental workload states (REST vs. high cognitive load) using the publicly available STEW dataset as a surrogate for the pitcher study data collection planned for next semester.

---

## Results (STEW Dataset — 48 subjects, real data)
| Model | Accuracy | F1 Score | ROC-AUC |
|---|---|---|---|
| SVM (RBF) | 0.922 | 0.922 | 0.977 |
| Random Forest | 0.863 | 0.865 | 0.939 |
| Logistic Regression | 0.823 | 0.822 | 0.900 |

---

## How to Run
**1. Download the STEW dataset** (free IEEE account required):
https://ieee-dataport.org/open-access/stew-simultaneous-task-eeg-workload-dataset

**2. Extract** all files into the same folder as `pipeline.py`

**3. Install dependencies:**
```bash
py -m pip install scipy scikit-learn matplotlib seaborn
```

**4. Run:**
```bash
py pipeline.py
```

Results are saved to a `results/` folder — includes figure, model scores CSV, and feature importance CSV.

---

## Pipeline Features
- Bandpass filtering + artifact rejection
- Spectral power extraction (theta, alpha, beta bands)
- Frontal θ / Parietal α ratio (PCLI proxy)
- Spectral entropy and temporal statistics
- SVM, Random Forest, Logistic Regression with 5-fold CV

---

## Connection to the HLR System
This pipeline directly implements the feature extraction and modeling workflow described in the research proposal. The frontal theta / parietal alpha ratio validated here will serve as the real-time **Pitcher Cognitive Load Index (PCLI)** input to the HLR model once pitcher EEG data is collected next semester using the GEOID HS500 HRV sensor.

---

## References
- Hassan, J., Reza, M. S., Ahmed, S. U., Anik, N. H., & Khan, M. O. (2025). EEG workload estimation and classification: a systematic review. Journal of Neural Engineering, 22(5), 051003. https://doi.org/10.1088/1741-2552/ad705e (doi.org in Bing)

- Lim, W. L., Sourina, O., & Wang, L. P. (2018). STEW: Simultaneous Task EEG Workload Data Set. IEEE Transactions on Neural Systems and Rehabilitation Engineering, 26(11), 2106–2114. https://doi.org/10.1109/TNSRE.2018.2872924 (doi.org in Bing)
