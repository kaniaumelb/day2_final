import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
from pathlib import Path
import hashlib
import json
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

os.environ["CHROMA_OPENAI_API_KEY"] = os.environ["OPENAI_API_KEY"]

CHUNKS_FOLDER = "chunks/Mohazab v Dick Smith Electronics Pty Ltd [No 2] (1995) 62 IR 200"
CASE_NAME = "Mohazab v Dick Smith Electronics Pty Ltd [No 2] (1995) 62 IR 200"

@st.cache_resource
def get_collection():
    client = chromadb.PersistentClient(path="./my_chroma_db")

    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        model_name="text-embedding-3-large",
    )

    collection = client.get_or_create_collection(
        name="knowledge_base",
        embedding_function=openai_ef,
    )

    folder = Path(CHUNKS_FOLDER)
    if folder.exists():
        files = sorted(folder.glob("chunk_*.json"))
        documents, metadatas, ids = [], [], []

        for file in files:
            with open(file, "r", encoding="utf-8", errors="ignore") as f:
                data = json.load(f)

            # try common key names for the chunk's actual text
            text = None
            for key in ("text", "content", "chunk_text", "page_content", "body"):
                if key in data and data[key]:
                    text = data[key].strip()
                    break

            if not text:
                st.warning(f"Couldn't find text field in {file.name} — check its keys: {list(data.keys())}")
                continue

            meta = {"source": CASE_NAME, "filename": file.name}
            # carry through any extra metadata fields that aren't the main text
            for key, value in data.items():
                if key not in ("text", "content", "chunk_text", "page_content", "body") and isinstance(value, (str, int, float)):
                    meta[key] = value

            documents.append(text)
            metadatas.append(meta)
            ids.append(hashlib.md5(file.name.encode()).hexdigest())

        if documents:
            collection.upsert(documents=documents, metadatas=metadatas, ids=ids)

    return collection

collection = get_collection()

# --- Search UI ---
st.title(f"{CASE_NAME.split('[')[0].strip()} — Search")
query = st.text_input("Search query")
n_results = 1

client = OpenAI()

if st.button("Search") and query:
    results = collection.query(query_texts=[query], n_results=n_results)
    for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
        with st.container(border=True):
            st.write(doc)
            st.caption(f"source: {meta.get('filename')} · distance: {dist:.3f}")

    context = "\n\n---\n\n".join(results["documents"][0])

    prompt = f"""You are a legal research assistant. Answer the question using only the context below.
    If the context doesn't contain the answer, say so — do not make anything up.

    Context:
    {context}

    Question: {query}

    Answer:"""

    with st.spinner("Generating response..."):
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )

    st.subheader("Answer")
    st.write(response.choices[0].message.content)