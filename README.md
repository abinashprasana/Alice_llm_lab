
# Alice LLM Lab

A small transformer based language model trained on *Alice’s Adventures in Wonderland*.  
This project was built as a hands on learning exercise to understand how language models work end to end, starting from raw text, moving through training a custom transformer, and finally experimenting with retrieval based context and interactive inference.

The entire setup runs locally and focuses on clarity, experimentation, and understanding rather than production scale performance.

---

## What this project is

Alice LLM Lab is a learning focused project that lets you:

• Train a tiny character level transformer from scratch  
• Generate text in the style of the training data  
• Experiment with retrieval augmented generation using a local SQLite database  
• Interact with the model through a simple Streamlit interface  

The goal is not to build a large or state of the art model, but to clearly understand each moving part in a modern LLM style pipeline.

---

## How it works (high level)

Think of the system as a simple flow:

1. Load and clean the raw Alice in Wonderland text  
2. Split the text into smaller chunks for training and retrieval  
3. Train a character level transformer on the processed text  
4. Store text chunks in SQLite for retrieval  
5. Retrieve relevant chunks when a prompt is given  
6. Use the retrieved context to guide text generation  

This keeps generated outputs grounded in the original text when retrieval is enabled.

---

## Project structure

```
alice-mini-llm/
│
├── app/
│   └── streamlit_app.py        # Streamlit interface for text generation
│
├── data/
│   ├── raw/
│   │   └── alice.txt           # Original raw text
│   ├── processed/
│   │   ├── alice_clean.txt     # Cleaned text
│   │   ├── chunks.jsonl        # Text chunks for retrieval
│   │   ├── train.txt           # Training data
│   │   └── val.txt             # Validation data
│   └── texts.db                # SQLite database for RAG
│
├── outputs/
│   ├── checkpoints/
│   │   ├── model.pt
│   │   ├── model_best.pt
│   │   └── model_latest.pt
│   └── plots/
│       └── loss.png            # Training loss curve
│
├── src/
│   ├── data_prep/
│   │   └── dataset_builder.py  # Data cleaning and preparation
│   │
│   ├── model/
│   │   ├── transformer.py      # Transformer architecture
│   │   ├── train.py            # Training logic
│   │   └── generate.py         # Text generation
│   │
│   ├── rag/
│   │   ├── retriever.py        # TF IDF based retrieval
│   │   └── rag_generate.py     # RAG based generation
│   │
│   ├── eval/
│   │   └── metrics.py          # Evaluation utilities
│   │
│   ├── config.py               # Central configuration
│   └── inference.py            # Shared inference helpers
│
├── Execution_Guide.md
├── Project_Report.md
├── requirements.txt
└── pyproject.toml
```

---

## Setup

### 1. Create a virtual environment (recommended)

```
python -m venv .venv
```

Activate it:

Windows (PowerShell):

```
.venv\Scripts\Activate.ps1
```

macOS / Linux:

```
source .venv/bin/activate
```

---

### 2. Install dependencies

```
pip install -r requirements.txt
```

---

## Preparing the dataset

Run the dataset builder to clean the raw text, create chunks, and prepare training files.

```
python src/data_prep/dataset_builder.py
```

This step also creates the SQLite database used for retrieval.

---

## Training the model

Start training the transformer using:

```
python src/model/train.py
```

Training outputs include model checkpoints and a loss curve saved under the outputs folder.

---

## Generating text

### Standard generation

```
python src/model/generate.py --prompt "Alice was beginning to"
```

### Retrieval augmented generation

```
python src/rag/rag_generate.py --prompt "Who is the Queen of Hearts?" --top_k 3
```

Retrieval uses TF IDF similarity over stored text chunks to provide relevant context.

---

## Running the Streamlit app

```
streamlit run app/streamlit_app.py
```

Once started, open the local URL shown in the terminal to interact with the model.

---

## Notes and limitations

• This is a learning focused prototype  
• The model is intentionally small and trained on limited data  
• Retrieval quality depends on chunking and TF IDF similarity  
• Performance depends on local hardware  

---

## Author

Abinash Prasana Selvanathan

If you find this project useful or interesting, feel free to star the repository.
