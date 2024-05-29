import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModel
import chromadb
import chromadb.config
from dotenv import load_dotenv
import torch
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Sidebar
with st.sidebar:
    st.title("LLM ChatApp")
    st.markdown('''
                ## About
                This app is an LLM powered ChatBot built using:
                - [Streamlit]
                - [LangChain]
                - [ChromaDB]
                - [Transformers]
                ''')
    add_vertical_space(5)
    st.write("Built by Joshua & Hassan")

load_dotenv()

# Initialize ChromaDB client
chroma_client = chromadb.Client(chromadb.config.Settings(persist_directory="./chromadb_store"))

# Initialize the Transformer model and tokenizer
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def embed_texts(texts):
    # Tokenize the texts
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    # Mean pooling to get sentence embeddings
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.tolist()

def create_collection_from_chunks(chunks, collection_name):
    # Embed the chunks
    embedded_chunks = embed_texts(chunks)
    
    # Create or get the collection
    collection = chroma_client.get_or_create_collection(name=collection_name)
    
    # Upsert the documents
    collection.upsert(
        documents=chunks,
        embeddings=embedded_chunks,
        ids=[f"{collection_name}_{i}" for i in range(len(chunks))]
    )
    return collection

def retrieve_data(query, collection_name):
    embedding = HuggingFaceEmbeddings(model_name=model_name)
    
    # Convert the query to an embedding
    query_embedding = embedding.embed_query(query)
    
    # Initialize the vector database
    vectordb = Chroma(embedding_function=HuggingFaceEmbeddings(model_name=model_name), client=chroma_client, collection_name=collection_name)
    
    # Retrieve the closest documents from the vector database
    search_results = vectordb.similarity_search(query_embedding, k=3)  # Retrieve top 3 similar documents
    
    if search_results:
        return [result.page_content for result in search_results]
    else:
        return ["No similar documents found."]

def main():
    st.title("LLM ChatApp")
    st.header("Chat with PDF")

    pdf = st.file_uploader("Upload PDF", type="pdf")
    if pdf is not None:
        pdf_reader = PyPDF2.PdfReader(pdf)
        text = "  "
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        
        store_name = pdf.name[:-4]
        
        # Create a collection from chunks
        create_collection_from_chunks(chunks, store_name)
        
        query = st.text_input("Enter your query")
        if query:
            results = retrieve_data(query, store_name)
            for result in results:
                st.write(result)

if __name__ == '__main__':
    main()
