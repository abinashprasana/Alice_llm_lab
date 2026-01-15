
# 🧠📚 Alice LLM Lab  
*A Tiny Transformer Language Model with Retrieval-Augmented Generation*

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Framework](https://img.shields.io/badge/UI-Streamlit-red)
![Model](https://img.shields.io/badge/Model-Custom%20Transformer-green)
![Status](https://img.shields.io/badge/Status-Learning%20Project-yellow)

---

## 📌 Project Overview

Alice LLM Lab is a **small transformer based language model** trained on *Alice’s Adventures in Wonderland*.  
This project was built as a **hands-on learning exercise** to understand how language models work end to end, starting from raw text, moving through training a custom transformer, and finally experimenting with retrieval based context and interactive inference.

Everything runs locally and the focus is on **clarity, experimentation, and understanding**, rather than production scale performance.

---

## 🎯 What this project does

This project allows you to:

📖 Train a tiny **character-level transformer** from scratch  
✍️ Generate text in the style of the training data  
🧩 Experiment with **retrieval-augmented generation** using a local SQLite database  
🖥️ Interact with the model through a **Streamlit-based interface**  

The goal is not to build a large model, but to clearly understand **each moving part** of a modern LLM-style pipeline.

---

## 🧠 How it works (high level)

Think of the system as a simple pipeline:

1️⃣ Load and clean the raw *Alice in Wonderland* text  
2️⃣ Split the text into smaller chunks for training and retrieval  
3️⃣ Train a character-level transformer on the processed text  
4️⃣ Store text chunks in a SQLite database  
5️⃣ Retrieve the most relevant chunks for a given prompt  
6️⃣ Use the retrieved context to guide text generation  

When retrieval is enabled, the generated output stays **grounded in the original text**.

---

## 🗂️ Project Structure (explained)

```
alice-mini-llm/
│
├── app/
│   └── streamlit_app.py        # Streamlit UI for generation
│
├── data/
│   ├── raw/
│   │   └── alice.txt           # Original raw text
│   ├── processed/
│   │   ├── alice_clean.txt     # Cleaned text
│   │   ├── chunks.jsonl        # Chunked text for retrieval
│   │   ├── train.txt           # Training split
│   │   └── val.txt             # Validation split
│   └── texts.db                # SQLite database for RAG
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
│   ├── eval/                   # Evaluation helpers
│   ├── config.py               # Central configuration
│   └── inference.py            # Shared inference helpers
│
├── Execution_Guide.md
├── Project_Report.md
├── requirements.txt
└── pyproject.toml
```

---

## ⚙️ Setup

### 1️⃣ Create a virtual environment (recommended)

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

Run the dataset builder to clean the raw text, create chunks, and prepare training files.

```
python src/data_prep/dataset_builder.py
```

This step also creates the SQLite database used for retrieval.

---

## 🏋️ Model Training

```
python src/model/train.py
```

Training outputs include:

📦 Model checkpoints  
📉 Training loss curve saved under `outputs/plots/`

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

Retrieval uses **TF-IDF similarity** over stored text chunks.

---

## 🖥️ Streamlit App

Run the interactive UI:

```
streamlit run app/streamlit_app.py
```

The app lets you:

📝 Enter prompts  
🎛️ Adjust generation parameters  
📚 Enable or disable retrieval  

---

## ⚠️ Notes & Limitations

✔️ This is a **learning-focused prototype**  
✔️ The model is intentionally small  
✔️ Retrieval quality depends on chunking and similarity scoring  
✔️ Performance depends on local hardware  

---

## 🙌 Author

**Abinash Prasana Selvanathan**  

⭐ If you found this project useful, feel free to star the repository.
