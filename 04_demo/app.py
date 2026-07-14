from __future__ import annotations

import os
import re
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import streamlit as st

from product_images import resolve_product_image

# wire up RAG module
_RAG_DIR = (Path(__file__).resolve().parent.parent / "03_rag").resolve()
sys.path.insert(0, str(_RAG_DIR))

# Module-level RAG imports (BUG-008: moved out of _ask() to avoid repeated
# import lookups on every chat message and to make deps visible to type checkers)
from src.retriever import retrieve  # type: ignore  # noqa: E402
from src.generator import generate, generate_raw  # type: ignore  # noqa: E402
from src.conversational import route_query, rewrite_query, simple_reply  # type: ignore  # noqa: E402
from src.web_retriever import web_retrieve  # type: ignore  # noqa: E402
from src.evaluator import groundedness  # type: ignore  # noqa: E402
from src.audit import log_audit_event  # type: ignore  # noqa: E402
from config.settings import settings  # type: ignore  # noqa: E402


# ── auto-ingestion on startup ──────────────────────────────────────────────────

def _check_and_run_ingest() -> None:
    # When launched from run_colab.py, ingestion runs in a separate cell before Streamlit.
    if os.environ.get("SKIP_INGEST") == "1":
        return
    if "ingest_checked" in st.session_state:
        return
    st.session_state["ingest_checked"] = True

    repo_dir = Path(__file__).resolve().parent.parent
    search_dirs = [
        repo_dir / "data" / "input",
        repo_dir / "Brand Resources",
    ]
    
    pdfs = []
    for d in search_dirs:
        if d.exists():
            pdfs.extend(d.rglob("*.pdf"))   # rglob: scan subdirectories too

            
    if not pdfs:
        return

    def _get_manifest_hashes(manifest_dir: Path) -> set[str]:
        hashes = set()
        if manifest_dir.exists():
            import json
            for f in manifest_dir.glob("*.json"):
                try:
                    data = json.loads(f.read_text(encoding="utf-8"))
                    if sha := data.get("sha256"):
                        hashes.add(sha)
                except Exception as e:
                    import logging
                    logging.warning(f"Failed to read manifest {f}: {e}")
        return hashes

    done_ocr = _get_manifest_hashes(repo_dir / "01_ocr" / "output" / "manifests")
    done_vlm = _get_manifest_hashes(repo_dir / "data" / "ocr_output_vlm" / "manifests")
    
    need_ingest = False
    import hashlib
    for pdf in pdfs:
        h = hashlib.sha256()
        with open(pdf, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        sha = h.hexdigest()
        
        if sha not in done_ocr or sha not in done_vlm:
            need_ingest = True
            break
            
    if need_ingest:
        with st.spinner("Processing new PDF catalogs (OCR + Figure indexing + Vector store re-indexing)..."):
            # Run ingest in a completely separate process to avoid BrokenPipeError.
            # When ingestion runs in-process, tqdm/HF flush Streamlit's hijacked stderr,
            # causing broken pipes. A subprocess gets its own clean streams.
            ingest_script = Path(__file__).resolve().parent.parent / "03_rag" / "ingest.py"
            env = os.environ.copy()
            env["TQDM_DISABLE"] = "1"
            env["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
            
            subprocess.run(
                [sys.executable, str(ingest_script)],
                env=env,
                check=True,
            )

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
    from src.generator import load_model  # type: ignore
    return load_model(_MODEL_ID)


@st.cache_resource(show_spinner="Loading index…")
def _load_index():
    from src.indexer import load_index  # type: ignore
    from src.retriever import load_reranker  # type: ignore
    index_dir = _RAG_DIR / "index"
    chunks_file = index_dir / "chunks.json"
    if not chunks_file.exists():
        # Index not built yet — ingestion hasn't run or is still running.
        # Return sentinel so callers can show a friendly message.
        return None, [], None
    client, chunks = load_index(index_dir)
    reranker = load_reranker()
    return client, chunks, reranker


# ── answer generation ─────────────────────────────────────────────────────────

_MODEL_ID = "Qwen/Qwen3-8B"


def _ask(question: str, summary: str) -> tuple[str, list]:
    model, tokenizer = _load_model()
    client, chunks, reranker = _load_index()

    if client is None:
        return (
            "⚠️ The product index is not ready yet — ingestion is still running or "
            "hasn't started. Please wait a few minutes, then refresh the page.",
            [],
        )

    # ── router: determine the route ──
    route = route_query(question, model, tokenizer, _MODEL_ID)
    
    if route == "NONE":
        return simple_reply(question, summary, model, tokenizer, _MODEL_ID), []

    # ── contextual rewrite: resolve "it / that / the same" into a standalone query ──
    retrieval_query = rewrite_query(question, summary, model, tokenizer, _MODEL_ID)

    if route == "WEB":
        st.toast("Searching the web for latest information...", icon="🌐")
        retrieved = web_retrieve(
            query=retrieval_query,
            reranker=reranker,
        )
    else:
        def hyde_fn(prompt: str, max_new_tokens: int = 120) -> str:
            return generate_raw(prompt, model, tokenizer, max_new_tokens=max_new_tokens)

        retrieved = retrieve(
            query=retrieval_query,
            client=client,
            chunks=chunks,
            reranker=reranker,
            generator_fn=hyde_fn if settings.hyde_enabled else None,
        )

    context_texts = [r.chunk.text for r in retrieved]

    # inject summary into context budget so history doesn't crowd out docs.
    # BUG-007: budget raised to 100000 chars (~25K tokens) to actually use the
    # 32K-token window now that the tokenizer max_length is fixed (BUG-002).
    sep = "\n\n---\n\n"
    summary_note = f"\nConversation so far:\n{summary}\n" if summary else ""
    n = max(len(context_texts), 1)
    sep_overhead = len(sep) * (len(context_texts) - 1) if context_texts else 0
    budget = (100_000 - len(summary_note) - sep_overhead) // n
    trimmed = [c[:budget] for c in context_texts]
    context = sep.join(trimmed) + summary_note

    # answer the original question (not the rewritten one) using retrieved context
    is_web = (route == "WEB")
    answer = generate(question, [context], model, tokenizer, _MODEL_ID, is_web=is_web)

    # soft grounding check: flag, never block
    if context_texts and not is_web and groundedness(answer, context_texts) < 0.45:
        answer += "\n\n_⚠️ Some details may not be fully covered in the catalog — please verify._"

    return answer, retrieved


# ── streamlit UI ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="MultiDimensions RAG",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

_check_and_run_ingest()

# init state
if "sessions" not in st.session_state:
    first = _new_session()
    st.session_state.sessions = {first.session_id: first}
    st.session_state.active_id = first.session_id

sessions: dict[str, Session] = st.session_state.sessions
active_id: str = st.session_state.active_id
# BUG-009: guard against stale active_id (e.g. two browser tabs desync)
if active_id not in sessions:
    active_id = next(reversed(sessions))
    st.session_state.active_id = active_id
active: Session = sessions[active_id]

# ── sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🔍 MultiDimensions RAG")
    st.caption("Industrial product catalog Q&A")
    st.divider()

    if st.button("＋ New Chat", use_container_width=True):
        s = _new_session()
        sessions[s.session_id] = s
        st.session_state.sessions = sessions
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
    # ARCH-002: show live index stats instead of hardcoded values
    try:
        _, _idx_chunks, _ = _load_index()
        _n_chunks = len(_idx_chunks)
        _n_docs = len({c.source_doc for c in _idx_chunks})
        st.caption(f"**Index:** {_n_chunks} chunks · {_n_docs} docs")
    except Exception:
        st.caption("**Index:** (loading…)")
    st.caption("**Retrieval:** SPLADE + dense + rerank")
    
    if st.button("Refresh Index", use_container_width=True, help="Click this after ingestion finishes to load the new data"):
        st.cache_resource.clear()
        st.rerun()

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

    # Show up to 4 images in a single row
    preview_limit = 4
    preview_images = images[:preview_limit]
    
    cols = st.columns(len(preview_images))
    for col, img in zip(cols, preview_images):
        caption = img.get("title", "Product image")
        source_doc = img.get("source_doc", "")
        if source_doc:
            caption = f"{caption}  ·  {source_doc}"
        with col:
            st.image(img["image_path"], caption=caption, use_container_width=True)

    # If there are more images, hide them inside a neat expander below
    if len(images) > preview_limit:
        remaining = images[preview_limit:]
        with st.expander(f"➕ View {len(remaining)} more images"):
            MAX_PER_ROW = 4
            for i in range(0, len(remaining), MAX_PER_ROW):
                row_images = remaining[i:i + MAX_PER_ROW]
                exp_cols = st.columns(len(row_images))
                for col, img in zip(exp_cols, row_images):
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
            if str(src['source_doc']).startswith("http"):
                st.markdown(f"🌐 **[{src['source_doc']}]({src['source_doc']})**")
            else:
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
            start_time = time.time()
            try:
                answer, retrieved = _ask(question, active.summary)
                success = True
                error_msg = ""
            except Exception as e:
                import traceback
                error_msg = traceback.format_exc()
                answer = "⚠️ I'm sorry, I encountered an internal error while processing your request. Please try asking again or refreshing the page."
                retrieved = []
                success = False
            
            elapsed_ms = int((time.time() - start_time) * 1000)

            # Extract route if possible (could be inferred or captured, for now we just log success/fail)
            # In a real app we'd pass route back from _ask, but here we just log the outcome.
            log_audit_event(
                user_id=active.user_id,
                session_id=active.session_id,
                question=question,
                route="UNKNOWN", # Route is hidden inside _ask, we could refactor to extract it later
                response_time_ms=elapsed_ms,
                success=success,
                error_message=str(e) if not success else "",
            )

        # The image resolver uses the raw answer to find <DISPLAY: ...> tags
        product_image = resolve_product_image(question, answer, retrieved)
        
        # Clean the answer for the UI so the user doesn't see the tags
        display_answer = re.sub(r"<DISPLAY:\s*[^>]+>", "", answer).strip()
        st.markdown(display_answer)
        
        _render_product_image(product_image)
        sources = _collect_sources(retrieved)
        _render_sources(sources)

    active.messages.append({
        "role": "assistant",
        "content": display_answer,
        "sources": sources,
        "product_image": product_image,
    })

    _update_summary(active, question, answer)
    st.session_state.sessions = sessions
    st.rerun()
