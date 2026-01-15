# Walkthrough - Alice Mini LLM Refactor

This document outlines the professional refactoring of the Alice Mini LLM project.

## Changes Implemented

### 1. Project Structure & Configuration
- **`pyproject.toml`**: The project is now an installable Python package.
- **`src/config.py`**: All paths, file names, and hyperparameters are defined in one place.
- **`src/eval/metrics.py`**: Renamed from `.coffee` to `.py`.

### 2. Core Logic
- **`src/model/transformer.py`**: Added docstrings and type hints.
- **`src/rag/retriever.py`**: encapsulated retrieval logic in `RAGPipeline`.
- **`src/inference.py`**: Created `ModelWrapper` to handle model loading (reused by App and CLI).

### 3. Application (UI)
- **`app/streamlit_app.py`**:
    -   Professionalized the tone (removed "student").
    -   Added an "How it Works" section for non-technical users.
    -   Renamed settings labels to be more intuitive (e.g., "Creativity" instead of just "Temperature").

## Execution Guide

### 1. Install Project
```powershell
pip install -e .
```

### 2. Prepare Data
```powershell
python -m src.data_prep.dataset_builder
```

### 3. Train the Model
Train for a longer duration to get good results:
```powershell
python -m src.model.train --max_steps 1000 --device cuda
```
*(Note: Remove `--device cuda` if you don't have a GPU. The script attempts to auto-detect, but explicit is better.)*

### 4. Run the Web App
```powershell
streamlit run app/streamlit_app.py
```

### 5. CLI Utility (Optional)
Generate text directly from the terminal:
```powershell
python -m src.rag.rag_generate --prompt "The Queen of Hearts" --top_k 3
```
