from __future__ import annotations

import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import streamlit as st

# wire up RAG module
_RAG_DIR = Path(__file__).parent.parent / "03_rag"
sys.path.insert(0, str(_RAG_DIR))

# ── session model ─────────────────────────────────────────────────────────────

@dataclass
class Session:
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    user_id: str = "demo_user"
    title: str = "New conversation"
    messages: list = field(default_factory=list)
    summary: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().strftime("%H:%M"))


def _new_session() -> Session:
    return Session()


def _update_summary(session: Session, question: str, answer: str) -> None:
    answer_short = answer.replace("\n", " ")[:150]
    line = f"• {question[:70]} → {answer_short}"
    lines = [l for l in session.summary.split("\n") if l.strip()]
    lines.append(line)
    session.summary = "\n".join(lines[-2:])  # keep last 2 turns, ~75 tokens max


def _set_title(session: Session, question: str) -> None:
    if session.title == "New conversation":
        session.title = question[:45] + ("…" if len(question) > 45 else "")


# ── RAG loading (once per app lifetime) ───────────────────────────────────────

@st.cache_resource(show_spinner="Loading model — this takes about 2 min on first run…")
def _load_model():
    from src.generator import load_model
    return load_model("Qwen/Qwen3-8B")


@st.cache_resource(show_spinner="Loading index…")
def _load_index():
    from src.indexer import load_index
    from src.retriever import load_reranker
    index_dir = _RAG_DIR / "index"
    collection, bm25, chunks = load_index(index_dir)
    reranker = load_reranker()
    return collection, bm25, chunks, reranker


# ── answer generation ─────────────────────────────────────────────────────────

def _ask(question: str, summary: str) -> tuple[str, list]:
    import torch
    from src.retriever import retrieve
    from src.generator import generate, generate_raw
    from config.settings import settings

    model, tokenizer = _load_model()
    collection, bm25, chunks, reranker = _load_index()

    def hyde_fn(prompt: str, max_new_tokens: int = 120) -> str:
        return generate_raw(prompt, model, tokenizer, max_new_tokens=max_new_tokens)

    retrieved = retrieve(
        query=question,
        collection=collection,
        bm25=bm25,
        chunks=chunks,
        reranker=reranker,
        generator_fn=hyde_fn if settings.hyde_enabled else None,
    )

    context_texts = [r.chunk.text for r in retrieved]

    # inject summary into context budget so history doesn't crowd out docs
    summary_note = f"\nConversation so far:\n{summary}\n" if summary else ""
    budget = (1800 - len(summary_note)) // max(len(context_texts), 1)
    trimmed = [c[:budget] for c in context_texts]
    context = "\n\n---\n\n".join(trimmed) + summary_note

    answer = generate(question, [context], model, tokenizer, "Qwen/Qwen3-8B")
    return answer, retrieved


# ── streamlit UI ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="MultiDimensions RAG",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# init state
if "sessions" not in st.session_state:
    first = _new_session()
    st.session_state.sessions = {first.session_id: first}
    st.session_state.active_id = first.session_id

sessions: dict[str, Session] = st.session_state.sessions
active_id: str = st.session_state.active_id
active: Session = sessions[active_id]

# ── sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🔍 MultiDimensions RAG")
    st.caption("Industrial product catalog Q&A")
    st.divider()

    if st.button("＋ New Chat", use_container_width=True):
        s = _new_session()
        sessions[s.session_id] = s
        st.session_state.active_id = s.session_id
        st.rerun()

    st.markdown("**Conversations**")
    for sid, sess in reversed(list(sessions.items())):
        label = ("▶ " if sid == active_id else "   ") + sess.title
        if st.button(label, key=f"sess_{sid}", use_container_width=True):
            st.session_state.active_id = sid
            st.rerun()

    st.divider()
    st.caption("**Model:** Qwen3-8B (4-bit)")
    st.caption("**Index:** 340 chunks · 18 docs")
    st.caption("**Retrieval:** BM25 + dense + rerank")

# ── main chat area ────────────────────────────────────────────────────────────

st.markdown(f"### {active.title}")

for msg in active.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Ask about the product catalog…")

if question:
    _set_title(active, question)
    active.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            answer, retrieved = _ask(question, active.summary)

        st.markdown(answer)

    active.messages.append({
        "role": "assistant",
        "content": answer,
    })

    _update_summary(active, question, answer)
    st.rerun()
