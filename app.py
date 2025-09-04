# app.py  —  RAG with Hybrid Retrieval + MMR + MultiQuery (Py 3.9 compatible)

import os
import json
import faiss
import streamlit as st

from typing import List, Tuple, Optional

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Sparse / hybrid bits
from langchain_community.retrievers import BM25Retriever
try:
    from langchain.retrievers import EnsembleRetriever
except Exception:
    from langchain.retrievers.ensemble import EnsembleRetriever  # older LC
try:
    from langchain.retrievers.multi_query import MultiQueryRetriever
except Exception:
    MultiQueryRetriever = None  # optional

# ---------------- Config ----------------
VECTORSTORE_DIR = "vectorstore/faiss_pdf_header_aware"
DEFAULT_CHAT_MODEL = "gpt-4o-mini"
FALLBACK_EMBED_BY_DIM = {3072: "text-embedding-3-large", 1536: "text-embedding-3-small"}

# --------------- API key load --------------
def load_api_key() -> Optional[str]:
    load_dotenv()  # .env support
    key = os.getenv("OPENAI_API_KEY")
    try:
        if not key and hasattr(st, "secrets"):
            key = st.secrets.get("OPENAI_API_KEY", None)
    except Exception:
        pass
    return key

OPENAI_API_KEY = load_api_key()

# --------------- UI setup ------------------
st.set_page_config(page_title="HOA Q&A", page_icon="🏠", initial_sidebar_state="collapsed")
st.title("🏠 HOA Q&A")
st.caption("Hi! This is a learning experiment to help neighbors answer questions about the HOA documents.")

with st.sidebar:
    with st.expander("⚙️ Advanced settings", expanded=False):
        st.subheader("Settings")
        k = st.slider("Chunks to retrieve (k)", 2, 20, 8)
        fetch_k = st.slider("FAISS fetch_k (MMR only)", 10, 100, 40, help="Candidate pool size for MMR.")
        temperature = st.slider("Answer creativity (temperature)", 0.0, 1.0, 0.1, 0.05)
        use_mmr = st.toggle("Use MMR (diversified dense)", value=True)
        use_hybrid = st.toggle("Use Hybrid (FAISS + BM25)", value=True)
        use_multiquery = st.toggle("Use Multi-Query Expansion", value=True, help="Requires extra LLM calls.")
        keyword_fallback = st.toggle("Keyword fallback scan", value=True, help="If dense retrieval looks weak, also scan chunks for literal matches.")
        quote_top = st.toggle("Show top chunk excerpt", value=False)
        st.markdown("---")
        st.markdown("**Vectorstore path:**")
        st.code(VECTORSTORE_DIR, language="bash")

if not OPENAI_API_KEY:
    st.error(
        "OPENAI_API_KEY is not set. Add it to a `.env` file, your environment, "
        "or Streamlit `secrets.toml` and rerun.\n\nExample `.env`:\n\nOPENAI_API_KEY=sk-..."
    )
    st.stop()

# --------------- Embedding model detection ---------------
def infer_embed_model_from_faiss(path: str) -> Tuple[str, int]:
    idx_path = os.path.join(path, "index.faiss")
    if not os.path.exists(idx_path):
        raise FileNotFoundError(f"FAISS index not found at: {idx_path}")
    index = faiss.read_index(idx_path)
    dim = index.d
    model = FALLBACK_EMBED_BY_DIM.get(dim)
    if not model:
        raise ValueError(
            f"Unknown FAISS index dimension {dim}. "
            "Update FALLBACK_EMBED_BY_DIM or rebuild the index with a known model."
        )
    return model, dim

def read_meta_model(path: str) -> Optional[str]:
    meta_path = os.path.join(path, "meta.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            return meta.get("embed_model")
        except Exception:
            return None
    return None

def resolve_embed_model(path: str) -> Tuple[str, int]:
    m = read_meta_model(path)
    if m:
        try:
            _, dim = infer_embed_model_from_faiss(path)
            return m, dim
        except Exception:
            return m, -1
    return infer_embed_model_from_faiss(path)

# --------------- Load vectorstore & build retrievers ---------------
@st.cache_resource(show_spinner="Loading vector index…")
def get_resources():
    embed_model, faiss_dim = resolve_embed_model(VECTORSTORE_DIR)

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model=embed_model)

    vs = FAISS.load_local(
        VECTORSTORE_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )

    # Try a quick dim sanity check
    try:
        test_vec = embeddings.embed_query("sanity check vector size")
        if len(test_vec) != vs.index.d:
            raise RuntimeError(
                f"Embedding model '{embed_model}' produced dim {len(test_vec)}, "
                f"but FAISS index is dim {vs.index.d}. Rebuild or switch models."
            )
    except Exception as e:
        st.warning(f"Embedding sanity check skipped/failed: {e}")

    # Build a BM25 index from the docstore (sparse retriever)
    all_docs = []
    try:
        for _id in vs.index_to_docstore_id.values():
            doc = vs.docstore.search(_id)
            if doc:
                all_docs.append(doc)
    except Exception as e:
        st.warning(f"Could not read all docs from FAISS docstore for BM25: {e}")

    bm25 = BM25Retriever.from_documents(all_docs) if all_docs else None

    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model=DEFAULT_CHAT_MODEL, temperature=0.0)

    return vs, bm25, llm, embeddings, embed_model, faiss_dim, all_docs

try:
    vs, bm25, llm, embeddings, EMBED_MODEL, EMBED_DIM, ALL_DOCS = get_resources()
except Exception as e:
    st.error(f"Failed to load vectorstore: {e}")
    st.stop()

def make_dense_retriever():
    if use_mmr:
        return vs.as_retriever(search_type="mmr", search_kwargs={"k": k, "fetch_k": max(fetch_k, k*3), "lambda_mult": 0.25})
    return vs.as_retriever(search_type="similarity", search_kwargs={"k": k})

def make_hybrid_retriever():
    dense = make_dense_retriever()
    if bm25 is None:
        return dense
    bm25.k = k  # ensure bm25 returns k docs
    return EnsembleRetriever(retrievers=[dense, bm25], weights=[0.6, 0.4])

# --------------- RAG chain -----------------
SYSTEM_PROMPT = """You are a careful HOA/RCW assistant.
Answer the user's question **using only** the provided context.
If the answer is not fully contained in the context, say you don't have enough information.
Cite sources as [source_name] at the end of the relevant sentence(s).
Be concise and specific; quote exact language when the question is about rules or definitions.
At the end, always state: This is AI-generated guidance and not an official answer from the NTHOA. For an official answer to your question
please send an email to nachesterracehoa@gmail.com. Thank you and have a nice day!"""

prompt = ChatPromptTemplate.from_messages(
    [("system", SYSTEM_PROMPT),
     ("human", "Question: {question}\n\nContext:\n{context}\n\nAnswer:")]
)
parser = StrOutputParser()

def format_context(docs: List) -> str:
    lines = []
    for i, d in enumerate(docs, 1):
        src = d.metadata.get("source", f"chunk_{i}")
        lines.append(f"[{src}] {d.page_content}")
    return "\n\n".join(lines)

def keyword_scan(query: str, max_hits: int = 20):
    """Simple substring scan over all chunks for sanity (case-insensitive)."""
    q = query.lower()
    hits = []
    for d in ALL_DOCS or []:
        text = d.page_content or ""
        if q in text.lower():
            hits.append(d)
            if len(hits) >= max_hits:
                break
    return hits

def retrieve_docs(question: str) -> List:
    # Build chosen retriever
    retr = make_hybrid_retriever() if use_hybrid else make_dense_retriever()

    # Optionally expand query with MultiQuery
    if use_multiquery and MultiQueryRetriever is not None:
        try:
            mqr = MultiQueryRetriever.from_llm(retriever=retr, llm=llm)
            docs = mqr.get_relevant_documents(question)
        except Exception:
            docs = retr.get_relevant_documents(question)
    else:
        docs = retr.get_relevant_documents(question)

    # If dense/hybrid retrieval looks weak, optionally do a keyword fallback
    if keyword_fallback and len(docs) < max(2, k//2):
        kw_docs = keyword_scan(question, max_hits=k)
        # Merge, preferring original order, then unique by (source + first 40 chars)
        seen = set()
        merged = []
        for d in list(docs) + kw_docs:
            key = (d.metadata.get("source",""), d.page_content[:40])
            if key not in seen:
                merged.append(d)
                seen.add(key)
        docs = merged[:max(k, 6)]
    return docs

def run_rag(question: str, k: int, temperature: float) -> dict:
    docs = retrieve_docs(question)
    context = format_context(docs) if docs else ""
    chain = prompt | llm.bind(temperature=temperature) | parser
    answer = chain.invoke({"question": question, "context": context})
    return {"answer": answer, "docs": docs}

# --------------- Chat UI -------------------
# --------------- Chat UI -------------------
if "history" not in st.session_state:
    st.session_state.history = []

with st.form("ask_form", clear_on_submit=False):
    col1, col2 = st.columns([5, 1])
    with col1:
        user_q = st.text_input(
            "Ask a question",
            placeholder="What’s the rule about street parking or driveways?",
            label_visibility="collapsed",
            key="user_q",
        )
    with col2:
        # small top spacer so the button aligns with the text field
        #st.markdown("<div style='height: 0.0rem'></div>", unsafe_allow_html=True)
        ask = st.form_submit_button("Ask", type="primary", use_container_width=True)

clear_chat = st.button("Clear chat")

if clear_chat:
    st.session_state.history = []
    st.experimental_rerun()

if ask and user_q.strip():
    with st.spinner("Thinking…"):
        try:
            result = run_rag(user_q.strip(), k=k, temperature=temperature)
            st.session_state.history.append((user_q.strip(), result))
        except AssertionError as ae:
            st.error(
                "Embedding dimension mismatch detected.\n\n"
                "Rebuild your FAISS index with a pinned model "
                "(e.g., `text-embedding-3-large`) **and** load the same model in the app.\n"
                f"Details: {ae}"
            )
        except Exception as e:
            st.error(f"Query failed: {e}")


# --- Render answers (history) ---
for q, res in st.session_state.history[::-1]:
    st.markdown(f"**Q:** {q}")
    st.markdown(res["answer"])

    with st.expander("Retrieval debug: show sources / excerpts"):
        if not res["docs"]:
            st.markdown("_No documents retrieved._")
        else:
            for i, d in enumerate(res["docs"], 1):
                src = d.metadata.get("source", f"chunk_{i}")
                st.markdown(f"- **{src}**")
                if quote_top:
                    st.code((d.page_content or "")[:1500])
    st.divider()


# --------------- Footer info ----------------
with st.sidebar:
    st.markdown("**Resolved Embedding Model:**")
    st.code(f"{EMBED_MODEL}  ({EMBED_DIM}-D)", language="text")
    st.markdown("**Chat Model:**")
    st.code(DEFAULT_CHAT_MODEL, language="text")
