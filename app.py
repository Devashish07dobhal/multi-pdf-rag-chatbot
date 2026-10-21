"""Multi-PDF RAG Chatbot

Pipeline:
    PDFs -> extract text (pypdf) -> chunk (LangChain) -> embed (MiniLM)
         -> FAISS index -> retrieve (MMR / similarity) -> LLM answer with sources

The LLM backend is configurable (Hugging Face, or any OpenAI-compatible API such
as Groq, OpenRouter or a local Ollama server). See README.md for setup.
"""

import copy
import io
import os
import threading
from datetime import datetime, timezone

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from huggingface_hub import InferenceClient

# ===========================================================
# Configuration
# ============================================================

load_dotenv()

def get_config(key, default=None):
    """Read a setting from Streamlit secrets (cloud) first, then env / .env (local)."""
    try:
        if key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass  # no secrets file when running locally
    return os.getenv(key, default)

# LLM_BACKEND: "hf" -> Hugging Face; "groq" -> Groq; "openai" -> any
# OpenAI-compatible endpoint (OpenRouter, local Ollama, etc.).
LLM_BACKEND = get_config("LLM_BACKEND", "hf").strip().lower()
HF_TOKEN = get_config("HF_TOKEN")

if LLM_BACKEND == "groq":
    LLM_MODEL = get_config("LLM_MODEL", "llama-3.3-70b-versatile")
    LLM_BASE_URL = get_config("LLM_BASE_URL", "https://api.groq.com/openai/v1")
    # GROQ_API_KEY is the recommended Streamlit secret for Groq.
    LLM_API_KEY = get_config("GROQ_API_KEY") or get_config("LLM_API_KEY")
else:
    LLM_MODEL = get_config("LLM_MODEL", "CohereLabs/tiny-aya-global")
    LLM_BASE_URL = get_config("LLM_BASE_URL")
    LLM_API_KEY = get_config("LLM_API_KEY", "ollama")  # Ollama ignores the value

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

MAX_HISTORY_TURNS = 3  # previous Q/A pairs sent to the LLM for follow-ups

# Guard rails so a public demo cannot be overloaded or drain the API quota
MAX_FILES = int(get_config("MAX_FILES", 5))
MAX_TOTAL_MB = int(get_config("MAX_TOTAL_MB", 25))
MAX_PAGES = int(get_config("MAX_PAGES", 300))
MAX_QUESTIONS = int(get_config("MAX_QUESTIONS", 40))  # per browser session
# Shared by ALL visitors, resets daily (UTC). 0 = unlimited. Tune to your LLM quota.
MAX_DAILY_QUESTIONS = int(get_config("MAX_DAILY_QUESTIONS", 100))
# (chunk size, overlap, k, MMR) - sample answers are cached only for these defaults
DEFAULT_SETTINGS = (800, 100, 6, True)

# Built-in sample document (no file needed, so the demo always works)
SAMPLE_NAME = "ml_basics.pdf"
SAMPLE_QUESTIONS = [
    "What is overfitting and how can it be reduced?",
    "What is the difference between precision and recall?",
    "How does retrieval-augmented generation work?",
]
SAMPLE_PAGES = [
    """Machine Learning Basics: A Short Guide

1. Types of Machine Learning
Machine learning is the practice of building systems that learn patterns from data instead of following hand-written rules. There are three main types. Supervised learning trains a model on labelled examples, such as emails marked as spam or not spam, so it can predict labels for new data. Unsupervised learning finds structure in unlabelled data, for example grouping customers into clusters. Reinforcement learning trains an agent to take actions in an environment and learn from rewards and penalties.

2. Overfitting and Generalization
A model overfits when it memorizes the training data, including its noise, and then performs poorly on unseen data. The goal is generalization: good performance on new examples. Overfitting can be reduced by collecting more data, simplifying the model, applying regularization such as L1 or L2 penalties, using dropout in neural networks, and stopping training early. Cross-validation helps detect overfitting by evaluating the model on several different train and validation splits.

3. Evaluation Metrics
Accuracy is the share of predictions that are correct, but it can be misleading when classes are imbalanced. Precision measures how many of the predicted positives are truly positive, so low precision means many false positives. Recall measures how many of the actual positives were found, so low recall means many false negatives. The F1 score is the harmonic mean of precision and recall. ROC-AUC summarizes how well a classifier separates the classes across all decision thresholds. A confusion matrix shows true positives, false positives, true negatives, and false negatives in one table.

4. Retrieval-Augmented Generation (RAG)
Large language models can produce fluent but incorrect answers when they lack the right facts. Retrieval-augmented generation reduces this problem by giving the model relevant text at question time. First, documents are split into smaller chunks. Each chunk is converted into an embedding, a numeric vector that captures its meaning, and stored in a vector database such as FAISS. When a user asks a question, the question is embedded too, and the most similar chunks are retrieved. These retrieved chunks are placed in the prompt so the language model can answer using only the supplied context.

5. Chunking and Retrieval Choices
Chunk size is a trade-off. Small chunks are precise but may lose surrounding context, while large chunks keep context but can dilute the match with the question. A small overlap between neighbouring chunks prevents important sentences from being cut in half. Maximal marginal relevance, or MMR, is a retrieval method that balances relevance with diversity, so the retrieved chunks are not near-duplicates of each other. Showing the source document and page number next to each answer lets users verify the response.

This sample document is provided so the chatbot can be tried without uploading a file.""",
]

# Default test cases for the bundled sample document (editable in the UI)
DEFAULT_EVAL_CASES = [
    {
        "question": "What is overfitting and how can it be reduced?",
        "expected_terms": "overfitting, regularization, cross-validation",
    },
    {
        "question": "What is the difference between precision and recall?",
        "expected_terms": "precision, recall, false positives, false negatives",
    },
    {
        "question": "How does retrieval-augmented generation work?",
        "expected_terms": "chunks, embedding, vector database",
    },
    {
        "question": "What are the types of machine learning?",
        "expected_terms": "supervised, unsupervised, reinforcement",
    },
]

# set_page_config must be the first Streamlit rendering call
st.set_page_config(
    page_title="Multi-PDF RAG Chatbot",
    page_icon="📚",
    layout="wide",
)

if LLM_BACKEND == "hf" and not HF_TOKEN:
    st.error(
        "**HF_TOKEN is missing.** Add `HF_TOKEN=hf_...` to your `.env` file "
        "(local) or to the app's *Secrets* (Streamlit Cloud). "
        "To use Groq, set `LLM_BACKEND=groq` and add `GROQ_API_KEY` to Secrets."
    )
    st.stop()

if LLM_BACKEND in ("groq", "openai") and not LLM_BASE_URL:
    st.error(
        "**LLM_BASE_URL is missing.** Configure it in your `.env` file or "
        "Streamlit Secrets."
    )
    st.stop()

if LLM_BACKEND == "groq" and not LLM_API_KEY:
    st.error(
        "**GROQ_API_KEY is missing.** Add `GROQ_API_KEY` to Streamlit Cloud "
        "Secrets (or your local `.env` file)."
    )
    st.stop()

if LLM_BACKEND not in ("hf", "groq", "openai"):
    st.error("Unsupported `LLM_BACKEND`. Use `hf`, `groq`, or `openai`.")
    st.stop()

# Session state defaults
st.session_state.setdefault("messages", [])
st.session_state.setdefault("vector_store", None)
st.session_state.setdefault("processed_signature", None)
st.session_state.setdefault("source", None)  # "upload" or "sample"
st.session_state.setdefault("stats", None)
st.session_state.setdefault("question_count", 0)
st.session_state.setdefault("index_settings", None)  # (chunk size, overlap) used for the index


# ============================================================
# Cached resources (loaded once, shared across sessions)
# ============================================================

@st.cache_resource(show_spinner="Loading embedding model (first run only)...")
def load_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


@st.cache_resource
def get_client():
    # Both clients expose the same .chat.completions.create(...) interface,
    # so the generation code works unchanged with either one.
    if LLM_BACKEND in ("groq", "openai"):
        from openai import OpenAI

        return OpenAI(
            base_url=LLM_BASE_URL,
            api_key=LLM_API_KEY,
            timeout=45.0,
            max_retries=1,
        )
    return InferenceClient(provider="auto", api_key=HF_TOKEN)


# ============================================================
# Ingestion
# ============================================================

class NamedBytesIO(io.BytesIO):
    """In-memory file with .name and .size, like a Streamlit UploadedFile."""

    def __init__(self, data, name):
        super().__init__(data)
        self.name = name
        self.size = len(data)


def validate_uploads(files):
    """Return an error message if the upload exceeds the limits, else None."""
    if len(files) > MAX_FILES:
        return f"Please upload at most {MAX_FILES} PDFs at a time."
    total_mb = sum(f.size for f in files) / (1024 * 1024)
    if total_mb > MAX_TOTAL_MB:
        return (
            f"The files total {total_mb:.1f} MB; the limit is {MAX_TOTAL_MB} MB. "
            "Please upload smaller files."
        )
    return None


def extract_text_from_pdfs(pdf_files):
    """Return (documents, number_of_pages_with_no_text)."""
    documents = []
    empty_pages = 0
    total_pages = 0

    for pdf_file in pdf_files:
        try:
            reader = PdfReader(pdf_file)
            n_pages = len(reader.pages)
        except Exception as e:
            st.warning(f"Could not read {pdf_file.name}: {e}")
            continue

        total_pages += n_pages
        if total_pages > MAX_PAGES:
            st.warning(
                f"Skipped {pdf_file.name}: the {MAX_PAGES}-page limit would be exceeded."
            )
            total_pages -= n_pages
            continue

        for page_number, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception:
                text = ""

            if text.strip():
                documents.append(
                    {
                        "text": text,
                        "metadata": {
                            "source": pdf_file.name,
                            "page": page_number,
                        },
                    }
                )
            else:
                empty_pages += 1

    return documents, empty_pages


def build_vector_store(documents, chunk_size, chunk_overlap):
    """Split pages into chunks, embed them, and index them in FAISS."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.create_documents(
        [d["text"] for d in documents],
        metadatas=[d["metadata"] for d in documents],
    )

    vector_store = FAISS.from_documents(chunks, load_embeddings())
    return vector_store, len(chunks)


def process_files(files, source, chunk_size, chunk_overlap, documents=None):
    """Extract, chunk, embed and index the files. Returns True on success.

    `documents` can be passed in (already extracted) to skip PDF reading.
    """
    if documents is None:
        error = validate_uploads(files)
        if error:
            st.error(error)
            return False

    with st.spinner("Reading and indexing PDFs..."):
        try:
            empty_pages = 0
            if documents is None:
                documents, empty_pages = extract_text_from_pdfs(files)

            if not documents:
                st.error(
                    "No readable text found. Scanned/image-only PDFs need OCR first."
                )
                return False

            vector_store, n_chunks = build_vector_store(
                documents, chunk_size, chunk_overlap
            )
        except Exception as e:
            st.error(f"Error while processing PDFs: {e}")
            return False

    st.session_state.vector_store = vector_store
    st.session_state.source = source
    st.session_state.processed_signature = tuple((f.name, f.size) for f in files)
    st.session_state.messages = []  # new documents -> fresh chat
    st.session_state.question_count = 0
    st.session_state.index_settings = (chunk_size, chunk_overlap)
    st.session_state.stats = {
        "files": len(files),
        "pages": len(documents),
        "chunks": n_chunks,
        "empty_pages": empty_pages,
    }
    return True


def load_sample(chunk_size, chunk_overlap):
    """Index the built-in sample document (text is embedded in this file)."""
    documents = [
        {"text": text, "metadata": {"source": SAMPLE_NAME, "page": number}}
        for number, text in enumerate(SAMPLE_PAGES, start=1)
    ]
    placeholder_file = NamedBytesIO(b"", SAMPLE_NAME)
    return process_files(
        [placeholder_file], "sample", chunk_size, chunk_overlap, documents=documents
    )


# ============================================================
# Retrieval
# ============================================================

def retrieve(vector_store, question, k, use_mmr):
    if use_mmr:
        docs = vector_store.max_marginal_relevance_search(
            question, k=k, fetch_k=k * 5, lambda_mult=0.6
        )
    else:
        docs = vector_store.similarity_search(question, k=k)

    # Remove duplicate chunks
    unique_docs, seen = [], set()
    for doc in docs:
        text = doc.page_content.strip()
        if text and text not in seen:
            seen.add(text)
            unique_docs.append(doc)
    return unique_docs


def evaluate_retrieval(vector_store, test_cases, k, use_mmr):
    """Evaluate retrieval using manually specified expected keywords."""

    results = []

    for test in test_cases:
        question = test["question"]
        expected_terms = test["expected_terms"]

        # Use the same retrieval method as the chatbot
        docs = retrieve(vector_store, question, k, use_mmr)

        retrieved_text = " ".join(doc.page_content.lower() for doc in docs)

        matched_terms = [
            term for term in expected_terms if term.lower() in retrieved_text
        ]

        coverage = len(matched_terms) / len(expected_terms) if expected_terms else 0

        results.append(
            {
                "Question": question,
                "Retrieved chunks": len(docs),
                "Matched terms": f"{len(matched_terms)}/{len(expected_terms)}",
                "Keyword coverage": round(coverage * 100, 2),
                "Missing terms": ", ".join(
                    term for term in expected_terms if term not in matched_terms
                ),
            }
        )

    average_coverage = (
        sum(row["Keyword coverage"] for row in results) / len(results)
        if results
        else 0
    )

    return results, average_coverage


def build_context(docs):
    return "\n\n".join(
        f"Source: {d.metadata.get('source', 'Unknown')}\n"
        f"Page: {d.metadata.get('page', 'Unknown')}\n"
        f"Content:\n{d.page_content}"
        for d in docs
    )


def unique_sources(docs):
    seen, sources = set(), []
    for d in docs:
        key = (d.metadata.get("source", "Unknown"), d.metadata.get("page", "Unknown"))
        if key not in seen:
            seen.add(key)
            sources.append({"source": key[0], "page": key[1]})
    return sources


# ============================================================
# Demo protection: shared daily cap + cached sample answers
# ============================================================
# Both live in server memory, shared by all visitors. They reset whenever the
# app restarts or wakes from sleep, so treat them as best-effort protection.

@st.cache_resource
def get_usage_tracker():
    return {"lock": threading.Lock(), "day": None, "count": 0}


def today_utc():
    return datetime.now(timezone.utc).date().isoformat()


def try_consume_daily_quota():
    """Reserve one LLM question from today's shared budget. False if used up."""
    if MAX_DAILY_QUESTIONS <= 0:
        return True
    tracker = get_usage_tracker()
    with tracker["lock"]:
        today = today_utc()
        if tracker["day"] != today:
            tracker["day"], tracker["count"] = today, 0
        if tracker["count"] >= MAX_DAILY_QUESTIONS:
            return False
        tracker["count"] += 1
        return True


def release_daily_quota():
    """Give a reservation back (used when the LLM call failed outright)."""
    if MAX_DAILY_QUESTIONS <= 0:
        return
    tracker = get_usage_tracker()
    with tracker["lock"]:
        if tracker["day"] == today_utc() and tracker["count"] > 0:
            tracker["count"] -= 1


@st.cache_resource
def get_answer_cache():
    return {"lock": threading.Lock(), "data": {}}


def normalize_question(question):
    return " ".join(question.lower().split())


def cacheable_key(question, settings):
    """Cache key if this is a suggested sample question under default settings."""
    if st.session_state.source != "sample" or settings != DEFAULT_SETTINGS:
        return None
    key = normalize_question(question)
    suggested = {normalize_question(q) for q in SAMPLE_QUESTIONS}
    return key if key in suggested else None


def get_cached_message(key):
    cache = get_answer_cache()
    with cache["lock"]:
        message = cache["data"].get(key)
    if message is None:
        return None
    cached = copy.deepcopy(message)
    cached["cached"] = True
    return cached


def store_cached_message(key, message):
    """Keep the first good answer for a suggested question."""
    cache = get_answer_cache()
    with cache["lock"]:
        cache["data"].setdefault(key, copy.deepcopy(message))


# ============================================================
# Generation
# ============================================================

PROMPT_TEMPLATE = """You are a helpful document question-answering assistant.

Answer the question using ONLY the information contained in the provided context.

Rules:
1. Include all relevant information available in the context.
2. Do not invent facts or add unsupported information.
3. Explain the answer clearly, using headings and bullet points when helpful.
4. When the question covers several items (for example projects, topics or
   sections), cover each relevant item separately, including its purpose,
   details and results wherever these are present in the context.
5. If some requested information is missing, clearly say so.
6. If the context does not contain the answer at all, reply exactly:
   "I could not find the answer in the uploaded documents."
7. Avoid unnecessary repetition.

Context:
----------------
{context}
----------------

Question:
{question}

Answer:
"""


def get_history():
    """Last few Q/A turns, so the model can handle simple follow-ups."""
    recent = st.session_state.messages[-2 * MAX_HISTORY_TURNS:]
    return [{"role": m["role"], "content": m["content"]} for m in recent]


def build_messages(question, context, history):
    return history + [
        {
            "role": "user",
            "content": PROMPT_TEMPLATE.format(context=context, question=question),
        }
    ]


def generate_answer(question, context, history):
    """Single (non-streaming) completion."""
    response = get_client().chat.completions.create(
        model=LLM_MODEL,
        messages=build_messages(question, context, history),
        max_tokens=1024,
        temperature=0.2,
    )
    return (response.choices[0].message.content or "").strip()


def stream_answer(question, context, history):
    """Yield the answer token by token."""
    stream = get_client().chat.completions.create(
        model=LLM_MODEL,
        messages=build_messages(question, context, history),
        max_tokens=1024,
        temperature=0.2,
        stream=True,
    )
    for chunk in stream:
        if not chunk.choices:
            continue
        token = getattr(chunk.choices[0].delta, "content", None)
        if token:
            yield token


def generate_streaming(question, context, history, placeholder):
    """Stream the answer into `placeholder`; fall back to a normal call if needed.

    Returns (text, ok). ok is False if the answer was empty or interrupted.
    """
    full = ""
    ok = True
    try:
        for token in stream_answer(question, context, history):
            full += token
            placeholder.markdown(full + "▌")
    except Exception:
        if not full:
            # Streaming failed before the first token -> try a regular request
            full = generate_answer(question, context, history)
        else:
            full += "\n\n_⚠️ The response was interrupted._"
            ok = False

    if not full.strip():
        full = "The model returned an empty response. Please try again."
        ok = False
    placeholder.markdown(full)
    return full, ok


def friendly_error(exc):
    """Turn provider/network exceptions into a short, actionable message."""
    status = getattr(exc, "status_code", None) or getattr(
        getattr(exc, "response", None), "status_code", None
    )
    text = str(exc).lower()

    if status == 402 or "credit" in text or "quota" in text:
        return (
            "The LLM provider's free quota is used up. Try again later, or switch "
            "provider in the app settings (see README)."
        )
    if status == 429 or "rate limit" in text or "too many requests" in text:
        return "Rate limit reached - please wait a few seconds and try again."
    if status in (401, 403):
        return "The LLM API key was rejected. Check the key in your secrets/.env."
    if status == 404 or ("model" in text and "not found" in text):
        return f"The model `{LLM_MODEL}` was not found at the configured provider."
    if "connection" in text or "timed out" in text or "timeout" in text or "network" in text:
        return (
            "Could not connect to the LLM provider. Verify the provider URL, "
            "Streamlit Secrets, and network availability, then retry."
        )
    return f"Something went wrong: {exc}"


# ============================================================
# UI helpers
# ============================================================

APP_CSS = """
<style>
.block-container {padding-top: 2.2rem; max-width: 1100px;}
footer {visibility: hidden;}
#MainMenu {visibility: hidden;}

/* Hero banner */
.hero {
    background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
    padding: 1.6rem 1.8rem; border-radius: 18px; margin-bottom: 1.2rem;
    box-shadow: 0 8px 24px rgba(79, 70, 229, 0.25);
}
.hero h1 {margin: 0 !important; padding: 0 !important; font-size: 2rem !important; color: #fff !important;}
.hero p {margin: .4rem 0 0 0 !important; font-size: 1.02rem !important; opacity: .92; color: #fff !important;}
.hero .badges {margin-top: .9rem; display: flex; gap: .5rem; flex-wrap: wrap;}
.hero .badge {
    background: rgba(255,255,255,.18); border: 1px solid rgba(255,255,255,.35);
    padding: .2rem .7rem; border-radius: 999px; font-size: .8rem; color: #fff;
}

/* "How it works" cards */
.card {
    border: 1px solid rgba(128,128,128,.25); border-radius: 16px;
    padding: 1.1rem 1.2rem; height: 100%; background: rgba(79,70,229,.06);
}
.card h4 {margin: 0 0 .35rem 0 !important; padding: 0 !important;}
.card p {margin: 0 !important; opacity: .85; font-size: .93rem;}

/* Index statistics */
.stats {display: flex; gap: .6rem; flex-wrap: wrap; margin: .2rem 0 1rem 0;}
.pill {
    border: 1px solid rgba(128,128,128,.3); background: rgba(79,70,229,.08);
    padding: .3rem .8rem; border-radius: 999px; font-size: .88rem;
}

/* Chat bubbles, buttons, panels */
[data-testid="stChatMessage"] {
    border: 1px solid rgba(128,128,128,.2); border-radius: 16px;
    padding: .9rem 1rem; margin-bottom: .6rem;
}
.stButton > button {border-radius: 10px; font-weight: 600; width: 100%;}
[data-testid="stExpander"] {border-radius: 12px;}
[data-testid="stSidebar"] {border-right: 1px solid rgba(128,128,128,.2);}
</style>
"""


def inject_css():
    st.markdown(APP_CSS, unsafe_allow_html=True)


def render_hero():
    st.markdown(
        '<div class="hero"><h1>📚 Multi-PDF RAG Chatbot</h1>'
        "<p>Ask questions across your PDFs. Answers come only from their content, "
        "with page-level sources.</p>"
        '<div class="badges"><span class="badge">FAISS vector search</span>'
        '<span class="badge">MiniLM embeddings</span>'
        '<span class="badge">Streaming answers</span>'
        '<span class="badge">Source citations</span></div></div>',
        unsafe_allow_html=True,
    )


def render_extras(message):
    """Sources and retrieved chunks, shown under an assistant answer."""
    if message.get("cached"):
        st.caption("⚡ Instant answer - saved from an earlier run on the sample document.")

    if message.get("sources"):
        with st.expander("📖 Sources"):
            for s in message["sources"]:
                st.write(f"📄 **{s['source']}** — Page {s['page']}")

    if message.get("chunks"):
        with st.expander("🔍 Retrieved chunks"):
            for i, chunk in enumerate(message["chunks"], start=1):
                st.markdown(f"**Chunk {i}** — {chunk['source']}, page {chunk['page']}")
                st.write(chunk["text"])
                st.divider()


def render_landing(chunk_size, chunk_overlap):
    """Shown before any document has been processed."""
    st.markdown("### How it works")
    steps = [
        ("1️⃣ Upload", "Add one or more PDFs in the sidebar."),
        (
            "2️⃣ Process",
            "Pages are split into chunks, embedded, and indexed in a FAISS vector store.",
        ),
        (
            "3️⃣ Ask",
            "Your question retrieves the most relevant chunks, and the LLM answers "
            "using only those - with page-level sources.",
        ),
    ]
    for col, (title, text) in zip(st.columns(3), steps):
        col.markdown(
            f'<div class="card"><h4>{title}</h4><p>{text}</p></div>',
            unsafe_allow_html=True,
        )

    st.write("")
    if st.button("🚀 Try it with a sample document", type="primary"):
        if load_sample(chunk_size, chunk_overlap):
            st.rerun()

    st.caption(
        f"🔒 PDFs are processed in memory for your session only and are not stored. "
        f"Retrieved text snippets are sent to the configured LLM provider to write "
        f"answers. Limits: {MAX_FILES} PDFs, {MAX_TOTAL_MB} MB total, "
        f"{MAX_PAGES} pages."
    )


def render_evaluation(top_k, use_mmr):
    """Developer tool: check that retrieval finds the facts you expect."""
    with st.expander("📊 Retrieval evaluation (developer tool)"):
        st.caption(
            "For each question, list keywords that should appear in the retrieved "
            "chunks (comma-separated). Coverage = share of keywords found. "
            "The defaults match the bundled sample document - edit or add rows "
            "for your own PDFs."
        )
        edited = st.data_editor(
            pd.DataFrame(DEFAULT_EVAL_CASES),
            num_rows="dynamic",
            key="eval_cases",
        )

        if st.button("Run evaluation"):
            test_cases = []
            for row in edited.to_dict("records"):
                question = str(row.get("question") or "").strip()
                terms = [
                    t.strip()
                    for t in str(row.get("expected_terms") or "").split(",")
                    if t.strip()
                ]
                if question and terms:
                    test_cases.append({"question": question, "expected_terms": terms})

            if not test_cases:
                st.warning("Add at least one question with expected keywords.")
                return

            with st.spinner("Evaluating retrieval quality..."):
                results, average = evaluate_retrieval(
                    st.session_state.vector_store, test_cases, top_k, use_mmr
                )

            st.metric("Average keyword coverage", f"{average:.2f}%")
            st.dataframe(results, hide_index=True)


# ============================================================
# Sidebar: upload + settings
# ============================================================

with st.sidebar:
    st.header("📄 Documents")
    uploaded_files = st.file_uploader(
        "Upload your PDF files",
        type=["pdf"],
        accept_multiple_files=True,
    )

    with st.expander("⚙️ Advanced settings"):
        chunk_size = st.slider("Chunk size (characters)", 300, 1500, 800, step=100)
        chunk_overlap = st.slider("Chunk overlap", 0, 200, 100, step=25)
        top_k = st.slider("Chunks to retrieve (k)", 2, 8, 6)
        use_mmr = st.toggle("Diverse retrieval (MMR)", value=True)

    process_clicked = st.button(
        "🔄 Process PDFs", type="primary", disabled=not uploaded_files
    )
    sample_clicked = st.button("📎 Load sample document")

    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.session_state.question_count = 0
        st.rerun()

    st.divider()
    st.caption(f"🤖 LLM: `{LLM_MODEL}`")
    st.caption(f"🧠 Embeddings: `{EMBEDDING_MODEL.split('/')[-1]}`")
    


current_signature = (
    tuple((f.name, f.size) for f in uploaded_files) if uploaded_files else None
)

if process_clicked and uploaded_files:
    process_files(uploaded_files, "upload", chunk_size, chunk_overlap)

if sample_clicked:
    load_sample(chunk_size, chunk_overlap)


# ============================================================
# Main area
# ============================================================

inject_css()
render_hero()

if st.session_state.vector_store is None:
    render_landing(chunk_size, chunk_overlap)
    st.stop()

stats = st.session_state.stats
if stats:
    st.markdown(
        '<div class="stats">'
        f'<span class="pill">📄 {stats["files"]} file(s)</span>'
        f'<span class="pill">📃 {stats["pages"]} pages</span>'
        f'<span class="pill">🧩 {stats["chunks"]} chunks</span>'
        f'<span class="pill">🔎 {"MMR" if use_mmr else "Similarity"} · k={top_k}</span>'
        "</div>",
        unsafe_allow_html=True,
    )
    if stats["empty_pages"]:
        st.warning(
            f"{stats['empty_pages']} page(s) had no extractable text "
            "(likely scanned images) and were skipped."
        )

if (
    st.session_state.source == "upload"
    and current_signature != st.session_state.processed_signature
):
    st.warning(
        "The uploaded files changed since the last processing. "
        "Click **Process PDFs** to update the index."
    )

render_evaluation(top_k, use_mmr)

# Replay chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_extras(message)


# ============================================================
# Question answering
# ============================================================

suggestion_box = st.empty()
if st.session_state.source == "sample" and not st.session_state.messages:
    with suggestion_box.container():
        st.markdown("**💡 Try asking:**")
        for i, suggestion in enumerate(SAMPLE_QUESTIONS):
            if st.button(suggestion, key=f"suggestion_{i}"):
                st.session_state.pending_question = suggestion
                st.rerun()

question = st.chat_input("Ask a question about your documents...") or (
    st.session_state.pop("pending_question", None)
)

if question:
    suggestion_box.empty()

    index_settings = st.session_state.index_settings
    settings = (*index_settings, top_k, use_mmr) if index_settings else None
    cache_key = cacheable_key(question, settings)
    cached = get_cached_message(cache_key) if cache_key else None

    if cached is None and st.session_state.question_count >= MAX_QUESTIONS:
        st.info(
            f"This session reached the limit of {MAX_QUESTIONS} questions. "
            "Refresh the page to start a new session."
        )
    elif cached is None and not try_consume_daily_quota():
        st.info(
            "The demo has reached its daily question limit (it resets at midnight UTC). "
            "Please try again tomorrow, or run the app yourself - see the GitHub README."
        )
    else:
        history = get_history()  # taken before adding the new question
        st.session_state.messages.append({"role": "user", "content": question})

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            if cached is not None:
                st.markdown(cached["content"])
                assistant_message = cached
            else:
                st.session_state.question_count += 1
                docs = []
                llm_failed = False
                answer_ok = False
                placeholder = st.empty()
                try:
                    placeholder.markdown("_Searching the documents..._")
                    docs = retrieve(
                        st.session_state.vector_store, question, top_k, use_mmr
                    )

                    placeholder.markdown("_Thinking..._")
                    answer, answer_ok = generate_streaming(
                        question, build_context(docs), history, placeholder
                    )
                except Exception as e:
                    llm_failed = True
                    answer = f"⚠️ {friendly_error(e)}"
                    placeholder.markdown(answer)

                if llm_failed:
                    release_daily_quota()  # a failed call should not use up the shared budget

                assistant_message = {
                    "role": "assistant",
                    "content": answer,
                    "sources": unique_sources(docs),
                    "chunks": [
                        {
                            "source": d.metadata.get("source", "Unknown"),
                            "page": d.metadata.get("page", "Unknown"),
                            "text": d.page_content,
                        }
                        for d in docs
                    ],
                }

                # Cache only clean, first-turn answers to suggested questions
                if (
                    cache_key
                    and answer_ok
                    and not history
                    and "could not find the answer" not in answer.lower()
                ):
                    store_cached_message(cache_key, assistant_message)

            render_extras(assistant_message)

        st.session_state.messages.append(assistant_message)