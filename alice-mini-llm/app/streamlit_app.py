import sys
from pathlib import Path

# Add project root to sys.path to ensure 'src' module is found
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import streamlit as st
from src.inference import ModelWrapper
from src.rag.retriever import RAGPipeline


# Cache resources to avoid reloading on every interaction
@st.cache_resource
def get_model_wrapper():
    return ModelWrapper()


@st.cache_resource
def get_rag_pipeline():
    return RAGPipeline()


def main():
    st.set_page_config(page_title="Alice Mini LLM", layout="wide", page_icon="🐰")
    
    st.title("🐰 Alice Mini LLM")
    st.markdown("### A Custom AI trained on *Alice's Adventures in Wonderland*")

    with st.expander("🤔 How does this work?"):
        st.markdown(
            """
            This program is an **Artificial Intelligence** (Neural Network) that learned to read and write 
            by studying the book *Alice's Adventures in Wonderland* thousands of times.
            
            *   **Text Generation**: When you type a prompt, the AI guesses the next likely letters based on what it learned from Lewis Carroll's writing style.
            *   **RAG (Retrieval Augmented Generation)**: This is a fancy way of saying "Open Book Test". 
                If you enable it, the AI first looks up relevant paragraphs from the book (the "cheat sheet") 
                and uses them to write a better answer.
            """
        )

    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Settings")
        
        st.subheader("Text Generation")
        max_new_tokens = st.slider(
            "Length of Generated Text", 
            min_value=50, max_value=600, value=250, step=25,
            help="How many characters the AI should write."
        )
        temperature = st.slider(
            "Creativity (Temperature)", 
            min_value=0.2, max_value=1.5, value=1.0, step=0.1,
            help="Low = Predictable and repetitive. High = Creative but might talk nonsense."
        )
        
        st.divider()
        st.subheader("Knowledge Retrieval (RAG)")
        use_rag = st.toggle("Enable 'Open Book' Mode", value=True, help="Let the AI search the book for context.")
        top_k = st.slider("Number of Search Results", 1, 10, 3, 1)

    # Main Input
    st.write("#### Try it out!")
    prompt = st.text_area("Start a sentence (and let the AI finish it):", value="The Queen of Hearts shouted", height=100)

    if st.button("Generate", type="primary"):
        if not prompt.strip():
            st.warning("Please enter a prompt.")
            return

        status_container = st.status("Thinking...", expanded=True)
        
        try:
            with status_container:
                # Load resources
                st.write("Loading model...")
                wrapper = get_model_wrapper()
                
                final_prompt = prompt
                context_str = ""
                hits = []

                if use_rag:
                    st.write("Retrieving context...")
                    rag = get_rag_pipeline()
                    hits = rag.search(prompt, top_k=top_k)
                    
                    context_str = "\n\n---\n\n".join([h[2] for h in hits])
                    final_prompt = RAGPipeline.format_augmented_prompt(prompt, context_str)
                
                st.write("Generating text...")
                output = wrapper.generate(
                    final_prompt, 
                    max_new_tokens=max_new_tokens, 
                    temperature=temperature
                )
                status_container.update(label="Complete!", state="complete", expanded=False)

            # Display Results
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("Model Output")
                st.success(output)

            with col2:
                if use_rag and hits:
                    st.subheader("Retrieved Context")
                    for i, score, txt in hits:
                        with st.expander(f"Chunk {i} (Score: {score:.4f})"):
                            st.write(txt)
                elif use_rag:
                    st.info("No relevant context found.")

        except FileNotFoundError:
            st.error("Model checkpoint not found! Please run training first.")
        except Exception as e:
            st.error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
