import streamlit as st
from pypdf import PdfReader
from openai import OpenAI
from dotenv import load_dotenv
import os
import json

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

st.title("Exercise 2.1")

CHUNKS_DIR = "chunks"
os.makedirs(CHUNKS_DIR, exist_ok=True)


def summarize_text(text, model="gpt-4o-mini"):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You summarize documents clearly and concisely for a general audience."
            },
            {
                "role": "user",
                "content": f"Summarize the following document in a few short paragraphs:\n\n{text}"
            }
        ]
    )
    return response.choices[0].message.content


def chunk_by_topic(text, model="gpt-4o-mini"):
    response = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "You divide documents into logical topical sections. "
                    "For each section, provide a short title, a 1-2 sentence summary, "
                    "and the original text content belonging to that section (verbatim, "
                    "not paraphrased). Respond ONLY with JSON in this exact format: "
                    '{"sections": [{"title": "...", "summary": "...", "content": "..."}]}'
                )
            },
            {
                "role": "user",
                "content": text
            }
        ]
    )

    result = json.loads(response.choices[0].message.content)
    return result["sections"]


def save_chunks_to_disk(sections, source_filename, doc_summary):
    base_name = os.path.splitext(source_filename)[0]
    doc_folder = os.path.join(CHUNKS_DIR, base_name)
    os.makedirs(doc_folder, exist_ok=True)

    summary_path = os.path.join(doc_folder, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {"source_file": source_filename, "document_summary": doc_summary},
            f, indent=2, ensure_ascii=False
        )

    chunk_paths = []
    for i, section in enumerate(sections, start=1):
        chunk_path = os.path.join(doc_folder, f"chunk_{i}.json")
        with open(chunk_path, "w", encoding="utf-8") as f:
            json.dump(section, f, indent=2, ensure_ascii=False)
        chunk_paths.append(chunk_path)

    return doc_folder, chunk_paths


def list_saved_documents():
    """Returns a list of document folder names already saved in CHUNKS_DIR."""
    return sorted([
        name for name in os.listdir(CHUNKS_DIR)
        if os.path.isdir(os.path.join(CHUNKS_DIR, name))
    ])


def load_saved_document(doc_name):
    """Loads the summary and chunks for a previously saved document."""
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


def display_sections(summary_text, sections):
    st.subheader("Document Summary")
    st.write(summary_text)

    st.subheader(f"Split into {len(sections)} topic sections")
    for i, section in enumerate(sections):
        with st.expander(f"{i + 1}. {section['title']}"):
            st.markdown(f"**Summary:** {section['summary']}")
            st.write(section["content"])


# --- Sidebar: browse previously saved documents ---
st.sidebar.header("Previously saved documents")
saved_docs = list_saved_documents()

if saved_docs:
    selected_doc = st.sidebar.selectbox(
        "Choose a document to view",
        options=["-- none --"] + saved_docs
    )

    if selected_doc != "-- none --":
        summary_data, sections = load_saved_document(selected_doc)
        st.info(f"Showing saved chunks for **{selected_doc}**")
        display_sections(summary_data["document_summary"], sections)
else:
    st.sidebar.write("No saved documents yet.")


# --- Main flow: upload and process a new document ---
st.header("Upload a new document")

uploaded_file = st.file_uploader(
    "Choose a PDF",
    type=["pdf"]
)

if uploaded_file is not None:

    reader = PdfReader(uploaded_file)

    text = "\n".join(
        page.extract_text() or ""
        for page in reader.pages
    )

    with st.spinner("Summarizing document..."):
        summary = summarize_text(text)

    with st.spinner("Identifying topics and sections..."):
        sections = chunk_by_topic(text)

    display_sections(summary, sections)

    doc_folder, chunk_paths = save_chunks_to_disk(sections, uploaded_file.name, summary)

    st.success(f"Saved {len(chunk_paths)} chunks to `{doc_folder}`")