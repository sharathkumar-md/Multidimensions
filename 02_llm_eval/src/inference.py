from __future__ import annotations

import time

from loguru import logger

from src.models import ModelResponse

_SYSTEM_PROMPT = """\
You are a technical expert assistant for GGB bearing and bushing products.
Answer questions precisely based ONLY on the provided document context.
- Include specific numbers, units, and technical values exactly as they appear in the context.
- If the information is not present in the context, reply exactly: "Not found in the provided context."
- Keep answers concise and factual. Do not add information from outside the context.\
"""


def run_inference(
    model_name: str,
    question_id: str,
    question: str,
    context: str,
    source_doc: str,
    ollama_host: str = "http://localhost:11434",
) -> ModelResponse:
    import ollama

    client = ollama.Client(host=ollama_host)
    user_message = f"Context:\n{context}\n\nQuestion: {question}"

    t0 = time.perf_counter()
    response = client.chat(
        model=model_name,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        options={"temperature": 0.0, "seed": 42},
    )
    latency_ms = (time.perf_counter() - t0) * 1000

    answer = response.message.content.strip()
    logger.debug(
        "{model} | {qid} | {lat:.0f}ms | {chars} chars",
        model=model_name,
        qid=question_id,
        lat=latency_ms,
        chars=len(answer),
    )

    return ModelResponse(
        model_name=model_name,
        question_id=question_id,
        question=question,
        context_doc=source_doc,
        answer=answer,
        latency_ms=round(latency_ms, 1),
        prompt_tokens=response.prompt_eval_count or 0,
        completion_tokens=response.eval_count or 0,
    )
