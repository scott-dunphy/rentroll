#!/usr/bin/env python
# coding: utf-8

#from langchain_openai import ChatOpenAI
from langchain.chat_models import ChatOpenAI
from openai import OpenAI
import os
from pinecone import Pinecone
import streamlit as st
from langchain.chains import RetrievalQA
from langchain.document_loaders import UnstructuredPDFLoader, UnstructuredFileLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
import glob
import json
from tqdm.autonotebook import tqdm
from langchain.chains import ConversationalRetrievalChain


#loaders = [UnstructuredFileLoader(os.path.join(os.getcwd(),fn)) for fn in list(glob.glob("/Users/scottdunphy/Documents/ODCE/*.pdf"))]

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

llm = ChatOpenAI(model_name="gpt-3.5-turbo")

client = OpenAI()

pc_env = "gcp-starter"
pc_key = "6ddd683f-f012-4ae6-bd94-e30590812ca0"
index_name = 'rentrolls'

pc = Pinecone(api_key=pc_key)
index = pc.Index(index_name)

model_name = 'text-embedding-ada-002'

embed = OpenAIEmbeddings(
    model=model_name,
    
)

from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone

def chatcre(query):
    vectorstore = Pinecone(
    index, embed.embed_query, 'text'
)
    vectorstore.similarity_search(
    query,  # our search query
    k=3  # return 3 most relevant docs
)

    llm = ChatOpenAI(
        openai_api_key=os.environ["OPENAI_API_KEY"],
        model_name='gpt-3.5-turbo',
        temperature=0.0
    )
    
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )

    return qa.run(query)

st.set_page_config(page_title='ChatCRE')
st.title('ChatCRE')

query_input = st.text_input("Ask questions about the leases, property updates, and loan agreement.")
try:
    if query_input:
        write_value = chatcre(query_input)
        st.write(write_value)
    else:
        st.write("Try these prompts:")
        st.write("Is there anything concerning in the property updates?")
        st.write("Abstract the leases into a table.")
        
except:
    write_value = "Try these prompts: \n Is there anything concerning in the property updates? \n Abstract the leases into a table."
    st.write(write_value)
