import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

st.write("Aigoo")

st.title("Home")
st.write("Welcome to my app")

if st.button("Go to Page 1"):
    st.switch_page("pages/page1.py")

if st.button("Go to Page 2"):
    st.switch_page("pages/page2.py")

if st.button("Go to Page 3"):
    st.switch_page("pages/page3.py")