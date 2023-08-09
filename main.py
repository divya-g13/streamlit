import os

import sys

import openai

from langchain.chains import ConversationalRetrievalChain, RetrievalQA

from langchain.chat_models import ChatOpenAI

from langchain.document_loaders import DirectoryLoader, TextLoader

from langchain.embeddings import OpenAIEmbeddings

from langchain.indexes import VectorstoreIndexCreator

from langchain.indexes.vectorstore import VectorStoreIndexWrapper

from langchain.llms import OpenAI

from langchain.vectorstores import Chroma
import streamlit as st

#from constants import API_KEY

os.environ["OPENAI_API_KEY"] =  st.secrets["OPENAI_API_KEY"]

 

# Enable to save to disk & reuse the model (for repeated queries on the same data)

PERSIST = False

 
query = None

if len(sys.argv) > 1:

  query = sys.argv[1]

 

if PERSIST and os.path.exists("persist"):

  print("Reusing index...\n")

  vectorstore = Chroma(persist_directory="persist", embedding_function=OpenAIEmbeddings())

  index = VectorStoreIndexWrapper(vectorstore=vectorstore)

else:

  loader = TextLoader("water.txt") # Use this line if you only need data.txt

  #loader = DirectoryLoader(os.path.join(os.getcwd(), 'data'))

  if PERSIST:

    index = VectorstoreIndexCreator(vectorstore_kwargs={"persist_directory":"persist"}).from_loaders([loader])

  else:

    index = VectorstoreIndexCreator().from_loaders([loader])

 
llm = OpenAI(temperature=0,model_name='')
chain = ConversationalRetrievalChain.from_llm(

  llm=llm,

  retriever=index.vectorstore.as_retriever(search_kwargs={"k": 1}),

)
st.title("Schneider Helpline")
if "messages" not in st.session_state:
  st.session_state.messages = []
  
for message in st.session_state.messages:
  with st.chat_message(message["role"]):
    st.markdown(message["content"])
    
prompt = st.chat_input("type your question here")
if prompt:
  with st.chat_message("user"):
    st.markdown(prompt)
  st.session_state.messages.append({"role":"user", "content": prompt})
  response = index.query(prompt)
  if "I don't know" in response:
            response = "Sorry, I do not know the answer. Please contact your Schneider PSP for more details!"
  with st.chat_message("assistant"):
    st.markdown(response)
  st.session_state.messages.append({"role":"assistant", "content": response})
 
 
 
 
 
 
 

