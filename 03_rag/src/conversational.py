from __future__ import annotations

import torch

# ── shared chat helper ────────────────────────────────────────────────────────

def _chat(messages: list[dict], model, tokenizer, model_id: str, max_new_tokens: int = 64) -> str:
    kwargs: dict = {"tokenize": False, "add_generation_prompt": True}
    if "qwen3" in model_id.lower():
        kwargs["enable_thinking"] = False
    prompt_text = tokenizer.apply_chat_template(messages, **kwargs)
    inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_ids = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


# ── 1. router: does this turn need a catalog lookup? ─────────────────────────

_ROUTER_SYS = (
    "You decide whether a sales chatbot needs to look up the product catalog to answer "
    "a message. Answer with a single word: YES or NO.\n"
    "- Greetings, thanks, small talk, or meta questions about you -> NO.\n"
    "- Anything about a product, brand, industry, specification, or application -> YES.\n"
    "When unsure, answer YES."
)


def needs_retrieval(question: str, model, tokenizer, model_id: str) -> bool:
    messages = [
        {"role": "system", "content": _ROUTER_SYS},
        {"role": "user", "content": f"Message: {question}\nLookup needed (YES/NO):"},
    ]
    out = _chat(messages, model, tokenizer, model_id, max_new_tokens=4).lower()
    # default to retrieval unless it clearly says no
    return not out.strip().startswith("no")


# ── 2. contextual query rewriter: follow-up -> standalone question ───────────

_REWRITE_SYS = (
    "You rewrite a sales rep's latest message into a single standalone search query for a "
    "product catalog. Use the conversation so far to resolve references like 'it', 'that', "
    "'this one', or 'the same'. Include the specific product, brand, or industry being "
    "discussed. If the message is already standalone, return it unchanged. "
    "Output only the rewritten query, nothing else."
)


def rewrite_query(question: str, history: str, model, tokenizer, model_id: str) -> str:
    if not history.strip():
        return question
    messages = [
        {"role": "system", "content": _REWRITE_SYS},
        {"role": "user", "content": f"Conversation so far:\n{history}\n\nLatest message: {question}\n\nStandalone query:"},
    ]
    out = _chat(messages, model, tokenizer, model_id, max_new_tokens=64)
    out = out.strip().strip('"').strip()
    # guard against the model returning junk or refusing
    if not out or len(out) > 4 * len(question) + 200:
        return question
    return out


# ── 3. simple reply for the no-retrieval path ────────────────────────────────

_SIMPLE_SYS = (
    "You are a friendly assistant for industrial sales reps. Reply briefly and naturally to "
    "small talk or greetings. If the rep asks what you can do, say you help them find products "
    "to pitch, look up specifications, and prepare for customer visits using the product catalog."
)


def simple_reply(question: str, history: str, model, tokenizer, model_id: str) -> str:
    messages = [{"role": "system", "content": _SIMPLE_SYS}]
    if history.strip():
        messages.append({"role": "user", "content": f"(context)\n{history}"})
    messages.append({"role": "user", "content": question})
    return _chat(messages, model, tokenizer, model_id, max_new_tokens=128)
