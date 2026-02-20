# Disentangling Depression from Identity: A Domain-Adversarial Framework

This repository contains the official implementation of the paper: **"Disentangling Depression from Identity: A Domain-Adversarial Framework for Privacy-Preserving Personalized Speech Depression Detection."**



We expose a critical flaw in personalized depression detection models: the **"Longitudinal Trap" (Identity Leakage)**. When models are trained on historical patient data, they tend to "cheat" by memorizing speaker identities rather than learning genuine pathological cues. 

To solve this, we propose a **Domain-Adversarial Neural Network (DANN)**. By treating speaker identity as a nuisance variable, our model unlearns voiceprints while retaining true diagnostic features, ensuring both high clinical utility and stringent patient privacy.



## 📂 Repository Structure (Final Codebase)

The codebase consists of 8 core scripts, carefully organized to reproduce the experiments for both Scenario A (Screening) and Scenario B (Monitoring):

### 1. Data Preparation
* `experiment_generator.py`
  Generates the size-matched, unbiased training and testing splits for both Scenario A (Strict Cross-Subject) and Scenario B (Longitudinal).

### 2. Deep Feature Extractors (Linear Probing Baselines)
* `C_run_linear_probing_A.py`
  Extracts frozen features using Wav2Vec2 and performs Logistic Regression for Scenario A.
* `C_run_linear_probing_B.py`
  Performs the same extraction for Scenario B, revealing the inflated performance caused by identity leakage.

### 3. Proposed Method (Domain-Adversarial Neural Networks)
* `run_dann_scenario_A_v2.py`
  Applies DANN to Scenario A (Ablation study). Automatically generates t-SNE visualization plots showing feature space distributions.
* `run_dann_B_v2.py`
  **The core implementation.** Applies DANN to Scenario B, effectively disentangling depression features from speaker identity. Includes automatic t-SNE plotting for performance analysis.

### 4. Full Fine-Tuning Baselines (Transformer Unfrozen)
* `replicate_huang_scenario_A_v2.py`
  Replicates previous literature's full fine-tuning pipeline on Scenario A. Includes gradient checkpointing optimizations for A100 GPUs.
* `replicate_huang_scenario_B_v2.py`
  Replicates previous literature's full fine-tuning pipeline on Scenario B.

### 5. Privacy Evaluation
* `final_unify_probe.py`
  The explicit privacy probe script to quantitatively measure Speaker Accuracy (Identity Leakage) across different feature spaces.

---

## ⚙️ Quick Start

**1. Install Dependencies**
Ensure you have PyTorch, Torchaudio, Transformers, and Scikit-learn installed in your environment.

**2. Generate the Dataset Splits**
```bash
python experiment_generator.py
