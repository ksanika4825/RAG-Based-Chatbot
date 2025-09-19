import os
import google.generativeai as genai
from langchain.vectorstores import FAISS # This will be the vector database
from langchain_community.embeddings import HuggingFaceEmbeddings  # To perform word embedding
from langchain.text_splitter import RecursiveCharacterTextSplitter # This is for chunking
from pypdf import PdfReader
import faiss 
import streamlit as st
from pdfextractor import text_extractor_pdf

# Create the main page
st.title('RAG Based CHATBOT')
tips = ''' Follow the steps to use this application:
* Upload your PDF document in sidebar.
* Write your query and start chatting with the bot.'''
st.subheader(tips)


# Load PDF in sidebar
st.sidebar.title(':orange[UPLOAD YOUR DOCUMENT HERE (PDF only)]')
file_uploaded = st.sidebar.file_uploader('Upload file')
if file_uploaded:
    file_text = text_extractor_pdf(file_uploaded)

    # Step 1: Configure the model

    # Configure LLM
    key = os.getenv('GOOGLE_API_KEY')
    genai.configure(api_key='AIzaSyCA2uu5GaTNoNUXeelsTv0G4rgzzjRL_Zs')
    llm_model = genai.GenerativeModel('gemini-2.5-flash-lite')

    # Configure embedding model
    embedding_model = HuggingFaceEmbeddings(model_name='all-miniLM-L6-v2')

    # Step 2 : Chunking (Create Chunks)
    splitter = RecursiveCharacterTextSplitter(chunk_size = 800, chunk_overlap = 200)
    chunks = splitter.split_text(file_text)

    # Step 3 : Lets create FAISS Vector Store
    vector_store = FAISS.from_texts(chunks,embedding_model)

    # Step 4 : Configure retriever
    retriever = vector_store.as_retriever(search_kwargs = {'k':3})

    #Lets create a function that takes query and return the generated text.
    def generate_response(query):
        # Step 6 : Retrieval (R)
        retrieved_docs = retriever.get_relevant_documents(query=query)
        context = ''.join([doc.page_content for doc in retrieved_docs])

        # Step 7 : To write an augmented prompt (A)
        prompt = f''' You are a helpful assistant using RAG.
        Here is the context{context}

        The query asked by user is as follows = {query}'''

        # Step 8 : Generation (G)
        content = llm_model.generate_content(prompt)
        return content.text
        

    # lets create a chatbot in order to start the conversation
    # Initialize chat if there is no history.
    if 'history' not in st.session_state:
        st.session_state.history = []

    # Display History
    for msg in st.session_state.history:
        if msg['role'] == 'user':
            st.write(f":green[User]: {msg['text']}") 
        else:
            st.markdown(f":orange[Chatbot]: {msg['text']}") 


    # Input from the user(using streamlit form)
    with st.form('Chat Form',clear_on_submit=True):
        user_input = st.text_input('Enter your text here: ')
        send = st.form_submit_button('Send')

    # Start the Coversation and append the output and query in history.

    if user_input and send:
        st.session_state.history.append({'role':'user','text':user_input})
        model_output = generate_response(user_input)
        st.session_state.history.append({'role':'chatbot','text':model_output})

        st.rerun()







        




    


   

  


