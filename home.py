import os
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

st.title("Home")
st.write("Welcome to my app")

st.page_link("pages/page1.py", label="Go to Page 1")
st.page_link("pages/page2.py", label="Go to Page 2")