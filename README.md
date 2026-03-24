# AirStable-Argyrodite-Discovery

This repository contains the code for the paper:
**Artificial Intelligence-Assisted Dopant Discovery toward Air-Stable Argyrodite Sulfide Electrolytes**.

An AI-assisted research framework for discovering air-stable argyrodite sulfide electrolyte dopants, integrating composition-based formation-energy prediction, reaction ΔG screening, and literature-grounded RAG analysis to enable fast, evidence-driven candidate prioritization.

## Overview

The codebase includes three connected workflows:

- Formation energy prediction from composition formulas
- Reaction free-energy (Delta G) grid search based on model predictions
- Literature RAG pipeline for dopant pre-screening

## Repository Structure

- `config.py`: global configuration
- `data/`: formula parsing, data split, dataset preparation
- `models/`: transformer-based formation-energy model
- `training/`: model training scripts
- `inference/`: prediction, attention extraction, Delta G search
- `rag_presceening/`: PDF processing, vector database, RAG QA

## Environment Setup

Use Python 3.10 or newer.

### Option A: venv + pip

```bash
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Option B: Conda

```bash
conda env create -f environment.yml
conda activate formation_e
```

## Run

### 1. Prepare datasets

```bash
python -m data.dataset
```

Outputs:

- `data/train.csv`
- `data/val.csv`
- `data/test.csv`

### 2. Train the formation-energy model

```bash
python -m training.train
```

Checkpoints are saved in `checkpoints/`.

### 3. Run batch prediction

```bash
python -m inference.generate_predictions
```

Outputs:

- `inference/metal_binaries_predictions.csv`
- `inference/li6ps5cl_mfn_doping_predictions.csv`

### 4. Run Delta G grid search

```bash
python -m inference.delta_g_grid_search
```

Output:

- `inference/li6ps5cl_mfn_deltaG_results.csv`

### 5. Extract transformer attention (optional)

```bash
python -m inference.extract_attention Li6PS5Cl --output-dir outputs
```

## RAG Pipeline

### Build database and start interactive QA

```bash
python -m rag_presceening.main --mode full --force
```

Modes:

- `full`: process PDFs, build vector DB, then start QA
- `build`: process PDFs and build vector DB only
- `qa`: start QA only (requires existing vector DB)

### Batch QA

```bash
python -m rag_presceening.batch_qa
```

### Plot RAG figures

```bash
python -m rag_presceening.plot_rag_figures --no-show
```
