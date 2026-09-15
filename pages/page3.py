import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import numpy as np
import os
import json

load_dotenv()
client = OpenAI()

st.title("Exercise 2.3 — Ask Questions About Your Document")

CHUNKS_DIR = "chunks"


# ---------- Reuse: list & load saved documents (from page1.py) ----------

def list_saved_documents():
    if not os.path.exists(CHUNKS_DIR):
        return []
    return sorted([
        name for name in os.listdir(CHUNKS_DIR)
        if os.path.isdir(os.path.join(CHUNKS_DIR, name))
    ])


def load_saved_document(doc_name):
    doc_folder = os.path.join(CHUNKS_DIR, doc_name)

    summary_path = os.path.join(doc_folder, "summary.json")
    with open(summary_path, "r", encoding="utf-8") as f:
        summary_data = json.load(f)

    chunk_files = sorted(
        [f for f in os.listdir(doc_folder) if f.startswith("chunk_")],
        key=lambda x: int(x.split("_")[1].split(".")[0])
    )

    sections = []
    for chunk_file in chunk_files:
        with open(os.path.join(doc_folder, chunk_file), "r", encoding="utf-8") as f:
            sections.append(json.load(f))

    return summary_data, sections


# ---------- Reuse: embeddings & cosine similarity (from page2.py) ----------

def get_embedding(text, model="text-embedding-3-large"):
    text = text.replace("\n", " ")
    response = client.embeddings.create(
        model=model,
        input=text,
    )
    return response.data[0].embedding


def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# ---------- New: embed all chunks for a document (cached) ----------

def embed_sections(doc_name, sections):
    """Embeds each section's content once, caching results in session_state."""
    cache_key = f"embeddings_{doc_name}"

    if cache_key not in st.session_state:
        with st.spinner(f"Embedding {len(sections)} sections..."):
            embeddings = [get_embedding(section["content"]) for section in sections]
        st.session_state[cache_key] = embeddings

    return st.session_state[cache_key]


# ---------- New: retrieve the most relevant chunk(s) for a question ----------

def retrieve_relevant_sections(question, sections, section_embeddings, top_k=3):
    question_embedding = get_embedding(question)

    scored = [
        (cosine_similarity(question_embedding, emb), section)
        for emb, section in zip(section_embeddings, sections)
    ]
    scored.sort(key=lambda x: x[0], reverse=True)

    return scored[:top_k]


# ---------- New: answer the question using retrieved context ----------

def answer_question(question, top_sections, model="gpt-4o-mini"):
    context = "\n\n".join(
        f"[Section: {section['title']}]\n{section['content']}"
        for _, section in top_sections
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You answer questions using ONLY the provided document excerpts. "
                    "If the answer isn't in the excerpts, say so. "
                    "Cite which section(s) you used in your answer."
                )
            },
            {
                "role": "user",
                "content": f"Document excerpts:\n\n{context}\n\nQuestion: {question}"
            }
        ]
    )
    return response.choices[0].message.content


# ---------- Main app flow ----------

saved_docs = list_saved_documents()

if not saved_docs:
    st.warning("No saved documents found. Go to Exercise 2.1 and upload a PDF first.")
else:
    selected_doc = st.selectbox("Choose a document to ask questions about:", saved_docs)

    if selected_doc:
        summary_data, sections = load_saved_document(selected_doc)

        st.info(f"Loaded **{selected_doc}** — {len(sections)} sections")
        with st.expander("Document summary"):
            st.write(summary_data["document_summary"])

        section_embeddings = embed_sections(selected_doc, sections)

        question = st.text_input("Ask a question about this document:")

        if st.button("Get Answer") and question.strip():
            with st.spinner("Finding relevant sections..."):
                top_sections = retrieve_relevant_sections(question, sections, section_embeddings, top_k=3)

            with st.spinner("Generating answer..."):
                answer = answer_question(question, top_sections)

            st.subheader("Answer")
            st.write(answer)

            st.subheader("Sources used")
            for score, section in top_sections:
                with st.expander(f"{section['title']} (similarity: {score:.3f})"):
                    st.markdown(f"**Summary:** {section['summary']}")
                    st.write(section["content"])