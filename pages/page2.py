import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import numpy as np

load_dotenv()
client = OpenAI()

st.title("Exercise 2.2")

# 1. Allows the user to copy and paste two different texts.
text1 = st.text_area("Paste your first text here:", "")
text2 = st.text_area("Paste your second text here:", "")

# 2. Create an embedding for each text.
if text1 and text2:
    response1 = client.embeddings.create(
        model="text-embedding-3-large",
        input=text1,
    )
    embedding1 = response1.data[0].embedding

    response2 = client.embeddings.create(
        model="text-embedding-3-large",
        input=text2,
    )
    embedding2 = response2.data[0].embedding

    # 3. Display the cosine similarity of the two embeddings.
    a = np.array(embedding1)
    b = np.array(embedding2)
    cosine_similarity = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    st.write(f"Cosine similarity: {cosine_similarity}")