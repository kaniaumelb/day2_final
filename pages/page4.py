import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import numpy as np

load_dotenv()

st.title("Exercise 2.4")


# setup a vector database

# 1. allows the user to ask questions of the document from page

# text area for a user question

# 2. retrieves the most relevant part of the document and uses this to answer the question

# retrieve most relevant chunks from vector database

# 3. responds to the user citing what information was used from the document

# as a genAI using the enginereed context (user query + best chunk) to answer question