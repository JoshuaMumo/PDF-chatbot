import streamlit as st
from transformers import GPTNeoForCausalLM, GPT2Tokenizer, AutoModel
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
# Load model directly
from transformers import AutoModel
model = AutoModel.from_pretrained("nvidia/NV-Embed-v1", trust_remote_code=True)


# Sidebar information
with st.sidebar:
    st.title("LLM ChatApp")
    st.markdown('''
        ## About
        This app is an LLM powered ChatBot built using:
        - [Streamlit](https://streamlit.io/)
        - [LangChain](https://langchain.com/)
        - [ChromaDB](https://www.trychroma.com/)
        - [Transformers](https://huggingface.co/transformers/)
    ''')
    st.write("Built by Joshua & Hassan")

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Function to get embeddings for text chunks using nvidia/NV-Embed-v1
def get_nv_embed_embeddings(chunks):
    embedding_model = AutoModel.from_pretrained("nvidia/NV-Embed-v1")
    embeddings = embedding_model.encode(chunks)
    return embeddings

# Main function to run the app
def main():
    st.title("LLM ChatApp")
    st.header("Chat with PDF")

    pdf = st.file_uploader("Upload PDF", type="pdf")
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        chunks = get_text_chunks(text)

        # Get embeddings for the chunks using nvidia/NV-Embed-v1
        embeddings = get_nv_embed_embeddings(chunks)

        # Load generation model and tokenizer
        generation_model_name = "EleutherAI/gpt-neo-2.7B"
        generation_model = GPTNeoForCausalLM.from_pretrained(generation_model_name)
        generation_tokenizer = GPT2Tokenizer.from_pretrained(generation_model_name)

        query = st.text_input("Ask a question about the PDF")
        if query:
            # Get embedding for the query using nvidia/NV-Embed-v1
            embedding_model = AutoModel.from_pretrained("nvidia/NV-Embed-v1")
            query_embedding = embedding_model.encode([query])[0]

            # Find the most relevant chunk
            similarities = cosine_similarity([query_embedding], embeddings)
            most_relevant_chunk_idx = np.argmax(similarities)
            most_relevant_chunk = chunks[most_relevant_chunk_idx]

            # Generate a response using the most relevant chunk and the query
            prompt = f" {most_relevant_chunk}"
            inputs = generation_tokenizer.encode(prompt, return_tensors="pt")
            outputs = generation_model.generate(inputs, max_length=1024, num_return_sequences=1)
            answer = generation_tokenizer.decode(outputs[0], skip_special_tokens=True)
            st.write(answer)

if __name__ == '__main__':
    main()