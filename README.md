
# 🧠📚 Alice LLM Lab
*A Tiny Transformer Language Model with Retrieval-Augmented Generation*

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Used-EE4C2C?logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?logo=streamlit&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-Database-003B57?logo=sqlite&logoColor=white)
![ML](https://img.shields.io/badge/ML-Transformer_Model-4CAF50)
![Status](https://img.shields.io/badge/Status-Learning_Project-FBC02D)

---

## 📌 Project Overview

Alice LLM Lab is a **custom transformer-based language model** trained on *Alice’s Adventures in Wonderland*.  
The project was built as a **hands-on learning exercise** to understand how modern language models work internally, starting from raw text preprocessing, moving through model training, and finally extending generation with retrieval-based context.

Everything runs locally and the emphasis is on **learning core concepts clearly**, not on production-scale optimization.

---

## 🎯 What this project covers

This project focuses on understanding and implementing the following core ideas:

📖 Character-level language modeling  
🧠 Transformer architecture built from scratch  
🏋️ Model training and checkpointing  
✍️ Text generation with temperature control  
🧩 Retrieval-Augmented Generation using local data  
🗄️ SQL-based storage for text retrieval  
🖥️ Interactive inference using Streamlit  

Each part is implemented explicitly so the full pipeline is easy to trace and reason about.

---

## 🧠 Techniques and methods used

The project intentionally uses a small but important set of techniques that appear in real-world LLM systems:

🔹 **Custom Transformer Architecture**  
Multi-head self-attention and feed-forward layers implemented using PyTorch.

🔹 **Character-Level Tokenization**  
The model learns directly from raw characters rather than prebuilt tokenizers.

🔹 **Train / Validation Split**  
Clean separation of training and validation data to track learning behavior.

🔹 **Model Checkpointing**  
Best, latest, and final model checkpoints are saved during training.

🔹 **Loss Visualization**  
Training loss is tracked and plotted to understand convergence.

🔹 **Text Chunking**  
Processed text is split into overlapping chunks for retrieval.

🔹 **SQLite Database**  
All text chunks are stored in a local SQLite database for fast lookup.

🔹 **TF-IDF Similarity Search**  
Relevant text chunks are retrieved using TF-IDF based similarity scoring.

🔹 **Retrieval-Augmented Generation**  
Retrieved context is injected into the prompt to guide generation.

🔹 **Streamlit Interface**  
A lightweight UI for interactive prompting and experimentation.

---

## 🗂️ Project Structure (explained)

```
alice-mini-llm/
│
├── app/
│   └── streamlit_app.py        # Streamlit UI for inference
│
├── data/
│   ├── raw/
│   │   └── alice.txt           # Original dataset
│   ├── processed/
│   │   ├── alice_clean.txt     # Cleaned text
│   │   ├── chunks.jsonl        # Chunked text for retrieval
│   │   ├── train.txt           # Training data
│   │   └── val.txt             # Validation data
│   └── texts.db                # SQLite database
│
├── outputs/
│   ├── checkpoints/            # Model checkpoints
│   └── plots/
│       └── loss.png            # Training loss curve
│
├── src/
│   ├── data_prep/              # Dataset preparation
│   ├── model/                  # Transformer, training, generation
│   ├── rag/                    # Retrieval logic
│   ├── eval/                   # Metrics
│   ├── config.py               # Configuration
│   └── inference.py            # Shared inference helpers
│
├── Execution_Guide.md
├── Project_Report.md
├── requirements.txt
└── pyproject.toml
```

---

## ⚙️ Setup

### 1️⃣ Create a virtual environment

```
python -m venv .venv
```

Activate it:

**Windows (PowerShell)**

```
.venv\Scripts\Activate.ps1
```

**macOS / Linux**

```
source .venv/bin/activate
```

---

### 2️⃣ Install dependencies

```
pip install -r requirements.txt
```

---

## 🧪 Dataset Preparation

```
python src/data_prep/dataset_builder.py
```

This step cleans the text, creates chunks, builds the SQLite database, and prepares training files.

---

## 🏋️ Model Training

```
python src/model/train.py
```

Outputs include model checkpoints and a training loss plot.

---

## ✨ Text Generation

**Standard generation**

```
python src/model/generate.py --prompt "Alice was beginning to"
```

**Retrieval-augmented generation**

```
python src/rag/rag_generate.py --prompt "Who is the Queen of Hearts?" --top_k 3
```

---

## 🖥️ Streamlit App

```
streamlit run app/streamlit_app.py
```

Use the UI to experiment with prompts and retrieval settings.

---

## ⚠️ Notes and limitations

✔️ Learning-focused prototype  
✔️ Small model trained on limited data  
✔️ Retrieval quality depends on chunking and TF-IDF similarity  
✔️ Performance depends on local hardware  

---

## 🙌 Author

**Abinash Prasana Selvanathan**  

⭐ If you found this project useful, feel free to star the repository.
