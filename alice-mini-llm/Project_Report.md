# Final Project Report

## 1. Concept Verification
**Did I change the core concept?**
**No.** The heart of your project remains exactly the same.

*   **Original Goal**: Train a small AI model on *Alice in Wonderland* to generate text and use retrieval to answer questions.
*   **Current State**: Scaling a small AI model on *Alice in Wonderland* to generate text and use retrieval to answer questions.

**What DID change?**
I only changed **how** the code is organized, not **what** it does. Think of it like organizing a messy messy kitchen:
*   I put all the spices (settings) in one rack (`config.py`).
*   I labeled the jars (docstrings).
*   I fixed the broken stove instructions (fixed the `train.py` bugs).
*   I cleaned the dining table (improved the Streamlit UI).

The "recipe" (the math, the model architecture, the data) is untouched.

## 2. Comparison: Original vs. Refactored

| Feature | Original Code | Refactored Code | Why Change? |
| :--- | :--- | :--- | :--- |
| **Project Structure** | Flat files, script imports were broken without hacks. | Standard Python Package (`pyproject.toml`). | Allows `pip install -e .` so you can import code anywhere cleanly. |
| **Settings** | Hardcoded numbers (LR=3e-4) scattered in 5 files. | `src/config.py` | One file controls the whole project. Easier to experiment. |
| **Training** | `python train.py` (fixed settings). | `python -m src.model.train --args` | Flexible! You can change training steps from the command line. |
| **Model Loader** | Copy-pasted code in App and CLI. | `src/inference.py` (Class) | Write once, use everywhere. Easier to maintain. |
| **Retrieval (RAG)** | Loose functions. | `src/rag/retriever.py` (Class) | Cleaner logic, easier to read. |
| **UI (Streamlit)** | Basic inputs, "Student" labels. | Professional, "How it Works" help. | Friendly for non-programmers. |

## 3. Technical Concepts & Methods Used

This project covers several advanced AI and Software Engineering concepts:

### A. The AI Model (Large Language Model)
*   **Transformer Architecture**: The "brain" of the AI. We used a **Decoder-only Transformer** (like GPT-2, but tiny).
*   **Self-Attention**: The mechanism that allows the model to look at previous words to guess the next one.
*   **Character-Level Tokenization**: Instead of breaking text into words ("Alice", "was"), we break it into characters ("A", "l", "i", "c", "e"). This is simpler for small projects.
*   **Embeddings**: Converting letters into list of numbers (vectors) that the math can process.

### B. RAG (Retrieval Augmented Generation)
*   **TF-IDF (Term Frequency-Inverse Document Frequency)**: A statistical method to find "important" words. We use it to find paragraphs in the book that match your prompt.
*   **Cosine Similarity**: The math used to measure how similar two pieces of text are.
*   **Context Injection**: The process of taking the found paragraphs and secretly pasting them into the prompt so the AI can "read" them before answering.

### C. Software Engineering
*   **Modular Design**: Breaking code into small, focused files (`data`, `model`, `eval`) instead of one giant file.
*   **Argument Parsing (`argparse`)**: Creating professional command-line interfaces.
*   **Unit/Integration Testing**: (Implied) The verification steps we ran ensure the parts work together.
