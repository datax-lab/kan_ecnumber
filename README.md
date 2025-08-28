# Interpretable KANs for EC Number Prediction

This repository accompanies the paper on **Kolmogorov–Arnold Networks (KANs)** for Enzyme Commission (EC) number prediction. It integrates KAN layers into three representative backbones and provides utilities to evaluate classification at EC hierarchy levels 1–4 and to generate residue‑level interpretation maps.

## What this repository contains

* **Backbones with KAN variants**

  * DeepEC (convolutional encoder)
  * DeepECTransformer (attention‑based encoder)
  * CLEAN‑style classifier on top of protein language model embeddings
* **Interpretation utilities** that transform KAN spline responses into residue‑level importance scores suitable for motif‑like visualization.
* **Example notebook and checkpoint** to illustrate the interpretation workflow.

## How to use 

* Prepare protein sequences with EC annotations up to the desired hierarchy level. Deduplicate identical sequences and create train/validation/test splits with homology control when reproducing paper‑style results.
* Select a backbone (CNN, Transformer, or LLM‑embedding‑based) and its KAN variant to train and evaluate. Report micro‑ and macro‑F1 per EC level (1–4).
* Use the interpretation utilities to map learned responses back to residues and visualize important regions consistent with known motifs when available.

## Repository structure

* `DeepEC_KAN.py` — KAN‑augmented DeepEC implementation
* `DeepECtransformer_KAN.py` — KAN‑augmented Transformer implementation
* `CLEAN_KAN.py` — Classifier on top of PLM embeddings with KAN layers
* `Interpretation.py` — Functions for residue‑level interpretation and plotting helpers
* `Interpretation_test.ipynb` — Walkthrough of the interpretation pipeline
* `model_inter.ckpt` — Example checkpoint for interpretation

## Requirements

* Python == 3.9.19
* PyTorch == 2.2.2+cu118
* pykan == 0.2.4
* numpy == 1.26.23
* matplotlib == 3.9.0
* seaborn == 0.13.2
* efficient‑kan 

Additional common packages (for data handling, progress bars, or metrics) may be useful depending on your workflow.
