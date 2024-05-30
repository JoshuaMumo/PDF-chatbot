import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever

# Sidebar information
with st.sidebar:
    st.title("LLM ChatApp")
    st.markdown('''
        ## About
        This app is an LLM powered ChatBot built using:
        - [Streamlit](https://streamlit.io/)
        - [LangChain](https://github.com/langchain-ai/langchain)
        - [ChromaDB](https://www.trychroma.com/)
    ''')
    st.write("Built by Joshua & Hassan")
    def main():
         st.title("LLM ChatApp")
    st.header("Chat with PDF")

    pdf = st.file_uploader("Upload PDF", type="pdf")
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
               

        # Function to split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100,length_function=len)
        chunks = text_splitter.split_text(text)
       

        # Add to vector database
        vector_db = Chroma.from_texts(
            texts=chunks, 
            embedding=OllamaEmbeddings(model="nomic-embed-text", show_progress=True),
            collection_name="local-rag"
        )
        
        # Debug: Confirm vector database creation
        st.write("Vector database created and chunks embedded.")

        # LLM from Ollama
        local_model = "mistral"
        llm = ChatOllama(model=local_model)
        QUERY_PROMPT = PromptTemplate(
            input_variables=["question"],
            template="""You are an AI language model assistant. Your task is to generate five
            different versions of the given user question to retrieve relevant documents from
            a vector database. By generating multiple perspectives on the user question, your
            goal is to help the user overcome some of the limitations of the distance-based
            similarity search. Provide these alternative questions separated by newlines.
            Original question: {question}""",
        )
        retriever = MultiQueryRetriever.from_llm(
            vector_db.as_retriever(), 
            llm,
            prompt=QUERY_PROMPT
        )

        # RAG prompt
        template = """Answer the question based ONLY on the following context:
        {context}
        Question: {question}
        """

        prompt = ChatPromptTemplate.from_template(template)
        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        query = st.text_input("Ask a question about the PDF")
        if query:

            answer = chain.invoke({"question": query})
            
            # Debug: Print retrieved answer
            st.write(f"Retrieved answer: {answer}")

            st.write(answer)

if __name__ == '__main__':
    main()