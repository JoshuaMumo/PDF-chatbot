import streamlit as st
from transformers import GPTNeoForCausalLM, GPT2Tokenizer, AutoModel, AutoTokenizer
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import torch

# Sidebar information
with st.sidebar:
    st.title("LLM ChatApp")
    st.markdown('''
        ## About
        This app is an LLM powered ChatBot built using:
        - [Streamlit](https://streamlit.io/)
        - [LangChain](https://github.com/langchain-ai/langchain)
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

# Function to get embeddings for text chunks
def get_embeddings(chunks):
    embeddings = []
    embedding_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    embedding_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    for chunk in chunks:
        inputs = embedding_tokenizer(chunk, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            embedding = embedding_model(**inputs).last_hidden_state.mean(dim=1).squeeze().tolist()
        embeddings.append(embedding)
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
        embeddings = get_embeddings(chunks)

        generation_model_name = "EleutherAI/gpt-neo-125M"
        generation_model = GPTNeoForCausalLM.from_pretrained(generation_model_name)
        generation_tokenizer = GPT2Tokenizer.from_pretrained(generation_model_name)

        query = st.text_input("Ask a question about the PDF")
        if query:
            # Encode the query
            inputs = generation_tokenizer.encode(query, return_tensors="pt")

            # Generate a response from the model
            outputs = generation_model.generate(inputs, max_length=1024, num_return_sequences=1)
            answer = generation_tokenizer.decode(outputs[0], skip_special_tokens=True)
            st.write(answer)

if __name__ == '__main__':
    main()
