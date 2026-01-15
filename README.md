
# 🧠📚 Alice LLM Lab  
*A Tiny Transformer Language Model with Retrieval-Augmented Generation*

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Used-EE4C2C?logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?logo=streamlit&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-Database-003B57?logo=sqlite&logoColor=white)
![Status](https://img.shields.io/badge/Status-Learning_Project-FBC02D)

---

## 📌 Project Overview

Alice LLM Lab is a custom transformer-based language model trained on *Alice’s Adventures in Wonderland*.  
The project was built as a hands-on learning exercise to understand how modern language models work internally, starting from raw text preprocessing, moving through training a transformer from scratch, and extending generation using retrieval-based context.

The system uses a character-level modeling approach, explicit train and validation splits, checkpointed training, and loss tracking to make learning behavior visible.  
For grounding generation, processed text is chunked and stored in a local SQLite database, allowing relevant context to be retrieved and injected into prompts when required.

Everything runs locally and the focus is on clarity, experimentation, and understanding rather than production scale optimization.

---

## 🎯 What this project lets you do

• Train a small transformer language model from raw text  
• Observe training behavior through validation loss and checkpoints  
• Generate text in the style of the training corpus  
• Experiment with retrieval-augmented generation using local data  
• Interact with the model through a simple Streamlit interface  

---

## 🧠 How the system works (high level)

The project follows a clear and traceable flow:

1. Raw text is cleaned and normalized  
2. The processed text is split into training and validation data  
3. Text is chunked and stored in SQLite for retrieval  
4. A character-level transformer is trained using PyTorch  
5. Relevant text chunks are retrieved using TF-IDF similarity  
6. Retrieved context is optionally combined with prompts during generation  

---

<details>
<summary>🔗 <strong>How the programs are connected (click to expand)</strong></summary>

This project is designed so that each script feeds cleanly into the next, forming a complete and understandable pipeline.

**Dataset preparation**  
The flow begins with `dataset_builder.py`, which cleans the raw text, prepares training and validation splits, generates overlapping text chunks, and stores those chunks inside a SQLite database.

**Model training**  
`train.py` consumes the prepared training data and trains the transformer model. During training, checkpoints are saved and loss values are tracked so learning behavior can be inspected later.

**Standard generation**  
`generate.py` loads the trained model and produces text based only on the prompt and learned language patterns.

**Retrieval-augmented generation**  
When retrieval is enabled, `retriever.py` searches the SQLite database using TF-IDF similarity.  
`rag_generate.py` then injects the retrieved text into the prompt before passing it to the model.

**Interactive inference**  
`streamlit_app.py` ties everything together, allowing prompts, generation settings, and retrieval behavior to be controlled from a single interface.

Each stage is intentionally kept separate so individual components can be studied and modified without affecting the rest of the system.

</details>

---

<details>
<summary>📤 <strong>What outputs are generated (click to expand)</strong></summary>

Running the project produces several concrete outputs at different stages:

• **Processed text files** used for training and validation  
• **Chunked text data** stored for retrieval  
• **SQLite database** containing searchable text chunks  
• **Model checkpoints** saved during training  
• **Training loss plot** showing convergence behavior  
• **Generated text samples** produced via CLI or Streamlit  

These outputs make it easy to verify that each step in the pipeline is working as expected.

</details>

---

<details>
<summary>🗂️ <strong>Project structure (click to expand)</strong></summary>

```text
alice-mini-llm/
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   ├── raw/
│   │   └── alice.txt
│   ├── processed/
│   │   ├── alice_clean.txt
│   │   ├── chunks.jsonl
│   │   ├── train.txt
│   │   └── val.txt
│   └── texts.db
│
├── outputs/
│   ├── checkpoints/
│   │   ├── model.pt
│   │   ├── model_best.pt
│   │   └── model_latest.pt
│   └── plots/
│       └── loss.png
│
├── src/
│   ├── data_prep/
│   ├── model/
│   ├── rag/
│   ├── eval/
│   ├── config.py
│   └── inference.py
│
├── Execution_Guide.md
├── Project_Report.md
├── requirements.txt
└── pyproject.toml
```

</details>

---

<details>
<summary>⚙️ <strong>Setup and execution (click to expand)</strong></summary>

Create a virtual environment:
```
python -m venv .venv
```

Activate it:

Windows (PowerShell)
```
.venv\Scripts\Activate.ps1
```

macOS / Linux
```
source .venv/bin/activate
```

Install dependencies:
```
pip install -r requirements.txt
```

Prepare the dataset:
```
python src/data_prep/dataset_builder.py
```

Train the model:
```
python src/model/train.py
```

Generate text:
```
python src/model/generate.py --prompt "Alice was beginning to"
```

Run the Streamlit app:
```
streamlit run app/streamlit_app.py
```

</details>

---

## 🎥 Demo video

Add a short demo video here showing:
• Dataset preparation  
• Training progress  
• Text generation  
• Streamlit interaction  

Demo link:
```
<replace this with your demo video link>
```

---

<details>
<summary>⚠️ <strong>Notes and limitations (click to expand)</strong></summary>

• Learning-focused prototype  
• Small model trained on limited data  
• Retrieval quality depends on chunking and TF-IDF similarity  
• Performance depends on local hardware  

</details>

---

## 🙌 Author

**Abinash Prasana Selvanathan**  

⭐ If you found this project useful, feel free to star the repository.
