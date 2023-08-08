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

  # loader = TextLoader("data/data.txt") # Use this line if you only need data.txt

  loader = DirectoryLoader(os.path.join(os.getcwd(), 'data'))

  if PERSIST:

    index = VectorstoreIndexCreator(vectorstore_kwargs={"persist_directory":"persist"}).from_loaders([loader])

  else:

    index = VectorstoreIndexCreator().from_loaders([loader])

 
llm = OpenAI(temperature=0,model_name='')
chain = ConversationalRetrievalChain.from_llm(

  llm=llm,

  retriever=index.vectorstore.as_retriever(search_kwargs={"k": 1}),

)
#chat_history=[]
with st.form(key='my_form'):
    query = st.text_input("Prompt:", key='input_prompt')
    submit_button = st.form_submit_button(label='Ask')

 

    if query and submit_button:
        if query.lower() in ['quit', 'q', 'exit']:
            st.stop()
        #result = chain({"question": query, "chat_history": chat_history})
        response = index.query(query)
        
        st.write("Answer:", response)
        #chat_history.append((query, response))
 
 
 
 
 
 
 
def chatbot_prompt():
    print("Chatbot: Hi! I'm your friendly chatbot. How can I assist you today?")
    
    while True:
        user_input = input("You: ").strip().lower()
        
        if user_input == 'exit':
            print("Chatbot: Goodbye! Have a great day!")
            break
        
        
        response = index.query(user_input)
        print("Chatbot:", response)
 
 

#chatbot_prompt()
