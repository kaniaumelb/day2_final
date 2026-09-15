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

CHUNKS_ROOT = "chunks"  # parent folder containing one subfolder per case

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

    root = Path(CHUNKS_ROOT)
    if root.exists():
        documents, metadatas, ids = [], [], []

        for case_folder in sorted(p for p in root.iterdir() if p.is_dir()):
            case_name = case_folder.name

            for file in sorted(case_folder.glob("chunk_*.json")):
                with open(file, "r", encoding="utf-8", errors="ignore") as f:
                    data = json.load(f)

                text = None
                for key in ("text", "content", "chunk_text", "page_content", "body"):
                    if key in data and data[key]:
                        text = data[key].strip()
                        break

                if not text:
                    st.warning(f"Couldn't find text field in {case_folder.name}/{file.name} — keys: {list(data.keys())}")
                    continue

                meta = {"source": case_name, "filename": file.name}
                for key, value in data.items():
                    if key not in ("text", "content", "chunk_text", "page_content", "body") and isinstance(value, (str, int, float)):
                        meta[key] = value

                documents.append(text)
                metadatas.append(meta)
                ids.append(hashlib.md5(f"{case_name}/{file.name}".encode()).hexdigest())

        if documents:
            collection.upsert(documents=documents, metadatas=metadatas, ids=ids)

    return collection


def get_case_names():
    """List case folder names, used to populate the dropdown."""
    root = Path(CHUNKS_ROOT)
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir())


collection = get_collection()

# --- Search UI ---
st.title("Case Law Search")

case_names = get_case_names()
selected_case = st.selectbox("Select a case to search", case_names)

query = st.text_input("Search query")
n_results = 3

client = OpenAI()

if st.button("Search") and query and selected_case:
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        where={"source": selected_case},  # restrict search to the chosen case only
    )

    if not results["documents"][0]:
        st.info("No matching chunks found in this case.")
    else:
        for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
            with st.container(border=True):
                st.write(doc)
                st.caption(f"file: {meta.get('filename')} · distance: {dist:.3f}")

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