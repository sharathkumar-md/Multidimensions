from __future__ import annotations

import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import streamlit as st

from product_images import resolve_product_image

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

_MODEL_ID = "Qwen/Qwen3-8B"


def _ask(question: str, summary: str) -> tuple[str, list]:
    from src.retriever import retrieve
    from src.generator import generate, generate_raw
    from src.conversational import needs_retrieval, rewrite_query, simple_reply
    from src.evaluator import groundedness
    from config.settings import settings

    model, tokenizer = _load_model()
    collection, bm25, chunks, reranker = _load_index()

    # ── router: chit-chat / greetings skip the catalog lookup ──
    if not needs_retrieval(question, model, tokenizer, _MODEL_ID):
        return simple_reply(question, summary, model, tokenizer, _MODEL_ID), []

    # ── contextual rewrite: resolve "it / that / the same" into a standalone query ──
    search_query = rewrite_query(question, summary, model, tokenizer, _MODEL_ID)

    def hyde_fn(prompt: str, max_new_tokens: int = 120) -> str:
        return generate_raw(prompt, model, tokenizer, max_new_tokens=max_new_tokens)

    retrieved = retrieve(
        query=search_query,
        collection=collection,
        bm25=bm25,
        chunks=chunks,
        reranker=reranker,
        generator_fn=hyde_fn if settings.hyde_enabled else None,
    )

    context_texts = [r.chunk.text for r in retrieved]

    # inject summary into context budget so history doesn't crowd out docs.
    # reserve room for the separators too, otherwise generate()'s 2800-char clip
    # eats the tail of the summary appended at the end.
    sep = "\n\n---\n\n"
    summary_note = f"\nConversation so far:\n{summary}\n" if summary else ""
    n = max(len(context_texts), 1)
    sep_overhead = len(sep) * (len(context_texts) - 1) if context_texts else 0
    budget = (2800 - len(summary_note) - sep_overhead) // n
    trimmed = [c[:budget] for c in context_texts]
    context = sep.join(trimmed) + summary_note

    # answer the original question (not the rewritten one) using retrieved context
    answer = generate(question, [context], model, tokenizer, _MODEL_ID)

    # soft grounding check: flag, never block
    if context_texts and groundedness(answer, context_texts) < 0.45:
        answer += "\n\n_⚠️ Some details may not be fully covered in the catalog — please verify._"

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
    st.caption("**Index:** 472 chunks · 23 docs")
    st.caption("**Retrieval:** BM25 + dense + rerank")

# ── main chat area ────────────────────────────────────────────────────────────

def _render_product_image(product_image: dict | None) -> None:
    """Render 1-3 product images side by side.

    Accepts the dict returned by resolve_product_image():
        { "images": [{image_path, title, source_doc}, ...], "from_index": bool }
    """
    if not product_image:
        return

    images = product_image.get("images", [])
    if not images:
        return

    cols = st.columns(len(images))
    for col, img in zip(cols, images):
        caption = img.get("title", "Product image")
        source_doc = img.get("source_doc", "")
        if source_doc:
            caption = f"{caption}  ·  {source_doc}"
        with col:
            st.image(img["image_path"], caption=caption, use_container_width=True)


def _render_sources(sources: list) -> None:
    if not sources:
        return
    with st.expander(f"📄 Sources ({len(sources)})"):
        for src in sources:
            st.markdown(f"**{src['source_doc']}** — page {src['page_num']}")
            st.caption(src["snippet"])


def _collect_sources(retrieved) -> list:
    # dedupe by document + page so the same doc doesn't repeat
    seen, sources = set(), []
    for r in retrieved:
        key = (r.chunk.source_doc, r.chunk.page_num)
        if key in seen:
            continue
        seen.add(key)
        sources.append({
            "source_doc": r.chunk.source_doc,
            "page_num": r.chunk.page_num,
            "snippet": r.chunk.text[:200].strip() + "…",
        })
    return sources


st.markdown(f"### {active.title}")

for msg in active.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            _render_product_image(msg.get("product_image"))
            _render_sources(msg.get("sources", []))

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
        product_image = resolve_product_image(question, answer, retrieved)
        _render_product_image(product_image)
        sources = _collect_sources(retrieved)
        _render_sources(sources)

    active.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
        "product_image": product_image,
    })

    _update_summary(active, question, answer)
    st.rerun()
