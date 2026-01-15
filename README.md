
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
This project was built as a hands-on learning exercise to understand how language models work end to end, starting from raw text preparation, moving through training a transformer from scratch, and extending generation using retrieval-based context.

The model operates at a character level, uses explicit training and validation splits, and saves checkpoints and loss plots during training so learning behavior is visible.  
To keep generated responses grounded, processed text is chunked and stored in a local SQLite database, allowing relevant passages to be retrieved and injected into prompts when needed.

Everything runs locally and the focus is on clarity, experimentation, and understanding rather than production scale optimization.

---

## 📍 What this project is

Alice LLM Lab is a learning-focused project that lets you:

• Train a small transformer language model from raw text  
• Observe how training progresses through checkpoints and loss curves  
• Generate text in the style of the training corpus  
• Experiment with retrieval-augmented generation using local data  
• Interact with the model through a simple Streamlit interface  

> ℹ️ **Note**  
> The model is trained from scratch on the provided dataset and runs entirely on your local machine. No external model APIs are used.

---

## 🧠 How the system works

At a high level, the system follows a simple and traceable flow:

1. Raw text is cleaned and normalized  
2. The processed text is split into training and validation data  
3. Text is chunked and stored in SQLite for retrieval  
4. A character-level transformer is trained using PyTorch  
5. Relevant text chunks are retrieved using TF-IDF similarity  
6. Retrieved context is optionally combined with prompts during generation  

This design keeps generated output closely tied to the original dataset while still allowing creative variation.

---

## 🔗 How the programs are connected

The project is structured so that each script feeds cleanly into the next.

The pipeline begins with the dataset preparation script, which handles text cleaning, chunk creation, and database storage.  
The training script then consumes the prepared data and produces trained model checkpoints and loss visualizations.

During inference, standard generation loads the trained model directly, while retrieval-augmented generation first searches the SQLite database for relevant context before generating text.  
The Streamlit application acts as a thin layer on top of these components, providing a single interface to experiment with prompts, generation settings, and retrieval behavior.

---

## 📤 Outputs generated

Running the project produces the following outputs:

• Cleaned and processed text files for training and validation  
• Chunked text stored in a local SQLite database  
• Model checkpoints saved during training  
• A training loss plot showing convergence behavior  
• Generated text samples from both CLI and Streamlit  

These outputs make it easy to inspect each stage of the pipeline independently.

---

## 🗂️ Project structure

```
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

---

## ⚙️ Setup and execution

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

---

## 🎥 Demo video

You can add a short demo video here showing the dataset preparation, training process, and interactive generation.

Demo link:
```
<replace this with your demo video link>
```

---

## ⚠️ Notes and limitations

• Learning-focused prototype  
• Small model trained on limited data  
• Retrieval quality depends on chunking and TF-IDF similarity  
• Performance depends on local hardware  

---

## 🙌 Author

**Abinash Prasana Selvanathan**  

⭐ If you found this project useful, feel free to star the repository.
