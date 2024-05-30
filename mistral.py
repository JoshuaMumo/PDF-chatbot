import streamlit as st


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