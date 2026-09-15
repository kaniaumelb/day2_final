import os
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

st.markdown(
    """
    <style>
    .stApp {
        background-color: #ffc0cb;
    }

    .typewriter {
        font-size: 2.2em;
        font-weight: 700;
        color: #4a154b;
        white-space: nowrap;
        overflow: hidden;
        border-right: 3px solid #4a154b;
        width: 0;
        animation:
            typing 2.5s steps(20, end) forwards,
            blink 0.75s step-end infinite;
    }

    @keyframes typing {
        from { width: 0; }
        to { width: 17ch; }
    }

    @keyframes blink {
        50% { border-color: transparent; }
    }

    .subtitle {
        font-size: 1.1em;
        color: #6a1b6a;
        opacity: 0;
        animation: fadeIn 1s ease-in forwards;
        animation-delay: 2.6s;
    }

    .hint {
        font-size: 1em;
        color: #8a2f8a;
        opacity: 0;
        animation: fadeIn 1s ease-in forwards;
        animation-delay: 3.2s;
    }

    @keyframes fadeIn {
        to { opacity: 1; }
    }
    </style>

    <div class="typewriter">Welcome to my app</div>
    <div class="subtitle">Let's get started 🎉</div>
    <div class="hint">please select a page on the left ^_^</div>
    """,
    unsafe_allow_html=True
)

load_dotenv()