from __future__ import annotations

import torch

# ── shared chat helper ────────────────────────────────────────────────────────

def _chat(messages: list[dict], model, tokenizer, model_id: str, max_new_tokens: int = 64) -> str:
    kwargs: dict = {"tokenize": False, "add_generation_prompt": True}
    if "qwen3" in model_id.lower():
        kwargs["enable_thinking"] = False
    prompt_text = tokenizer.apply_chat_template(messages, **kwargs)
    # BUG-010: increased max_length so history never gets silently truncated
    inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
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


# ── 1. router: how should this query be handled? ───────────────────────────

_ROUTER_SYS = (
    "You decide how a sales chatbot should handle a user's message. Answer with a single word: LOCAL, WEB, or NONE.\n"
    "- Greetings, thanks, small talk, or meta questions about you -> NONE.\n"
    "- WARNING: If the question is about sports, entertainment, politics, recipes, or anything UNRELATED to industrial products, manufacturing, engineering, or B2B sales -> NONE.\n"
    "- Questions about general industry standards, competitor comparisons, or market trends not in our catalog -> WEB.\n"
    "- Questions about our products, brands, specifications, or applications -> LOCAL.\n"
    "When unsure about products vs web, default to LOCAL."
)


def route_query(question: str, model, tokenizer, model_id: str) -> str:
    messages = [
        {"role": "system", "content": _ROUTER_SYS},
        {"role": "user", "content": f"Message: {question}\nRouting decision (LOCAL/WEB/NONE):"},
    ]
    out = _chat(messages, model, tokenizer, model_id, max_new_tokens=6).lower()
    if "web" in out:
        return "WEB"
    elif "none" in out:
        return "NONE"
    return "LOCAL"


# ── 2. contextual query rewriter: follow-up -> standalone question ───────────

_REWRITE_SYS = (
    "You rewrite a sales rep's latest message into a single standalone search query for a "
    "product catalog. Resolve references like 'it', 'that', 'this one', 'them', or 'the same' "
    "using the conversation so far, carrying over the specific product, brand, or industry "
    "being discussed.\n"
    "Rules:\n"
    "- If the latest message is already a complete, standalone query, return it EXACTLY unchanged.\n"
    "- Never turn a specific question into a vaguer or more open-ended one.\n"
    "- Resolve every pronoun and 'the same'-style phrase into the concrete thing it refers to.\n"
    "- Output only the rewritten query, nothing else.\n"
    "Examples:\n"
    "Conversation: • what linear motors do you sell? → LinMot P01 series\n"
    "Latest: what's its stroke length?\n"
    "Query: what is the stroke length of the LinMot P01 linear motor\n"
    "---\n"
    "Conversation: • which products fit the packaging industry? → LinMot linear guides\n"
    "Latest: and what about the same for pharma?\n"
    "Query: which products fit the pharmaceutical industry\n"
    "---\n"
    "Conversation: (none)\n"
    "Latest: what linear motors do you sell?\n"
    "Query: what linear motors do you sell"
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
    "to pitch, look up specifications, and prepare for customer visits using the product catalog. "
    "If the user asks an out-of-scope question (e.g., sports, entertainment, politics, general trivia), "
    "politely remind them that you are a specialized industrial sales assistant and cannot answer questions outside that domain."
)


def simple_reply(question: str, history: str, model, tokenizer, model_id: str) -> str:
    # BUG-016: history belongs in the system prompt (background context), not
    # as a second user message — two consecutive user turns can confuse the
    # chat template and cause one of them to be ignored.
    sys_content = _SIMPLE_SYS
    if history.strip():
        sys_content = _SIMPLE_SYS + f"\n\nConversation summary:\n{history}"
    messages = [
        {"role": "system", "content": sys_content},
        {"role": "user", "content": question},
    ]
    return _chat(messages, model, tokenizer, model_id, max_new_tokens=128)
