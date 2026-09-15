import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import os
import numpy as np

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

st.title("Exercise 2.2 — Compare Two Texts")

# Initialize session state for both texts if they don't exist yet
if "text_1" not in st.session_state:
    st.session_state.text_1 = ""

if "text_2" not in st.session_state:
    st.session_state.text_2 = ""

# Text input widgets, bound to session state via `key`
text_1 = st.text_area(
    "Paste Text 1",
    value=st.session_state.text_1,
    height=250,
    key="text_1"
)

text_2 = st.text_area(
    "Paste Text 2",
    value=st.session_state.text_2,
    height=250,
    key="text_2"
)


def get_embedding(text, model="text-embedding-3-small"):
    response = client.embeddings.create(
        input=text,
        model=model
    )
    return response.data[0].embedding


def cosine_similarity(vec1, vec2):
    a = np.array(vec1)
    b = np.array(vec2)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


if st.button("Compare Texts"):
    if not st.session_state.text_1.strip() or not st.session_state.text_2.strip():
        st.warning("Please paste text into both boxes before comparing.")
    else:
        with st.spinner("Generating embeddings and comparing..."):
            embedding_1 = get_embedding(st.session_state.text_1)
            embedding_2 = get_embedding(st.session_state.text_2)
            similarity = cosine_similarity(embedding_1, embedding_2)

        st.subheader("Similarity Score")
        st.metric(label="Cosine Similarity", value=f"{similarity:.4f}")

        if similarity > 0.85:
            st.success("These texts are very similar.")
        elif similarity > 0.6:
            st.info("These texts are somewhat related.")
        else:
            st.warning("These texts appear to be quite different.")