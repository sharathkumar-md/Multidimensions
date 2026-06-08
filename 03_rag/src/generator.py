from __future__ import annotations

import re
import shutil
from pathlib import Path

import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

_THINK_RE = re.compile(r"<think>[\s\S]*?</think>", re.IGNORECASE)
# strip trailing hallucinated meta blocks (the model inventing its own follow-up Q&A)
_FOLLOWUP_RE = re.compile(
    r"\n\n(?:Question|Note|Q:|Follow-up|Additional question)\b.*",
    re.IGNORECASE | re.DOTALL,
)

_SYSTEM_PROMPT = (
    "You are a product specialist for an industrial and mechanical parts catalog, "
    "helping a sales representative answer a customer's question. "
    "Answer using only the provided catalog context.\n"
    "- Be technically precise: quote exact figures, units, model names, and material "
    "grades exactly as written (e.g. Nm, mm, °C, rpm, arc.min, AISI 316). Never invent, "
    "round, or estimate a specification.\n"
    "- Present multi-value or numeric specifications as a markdown table. Use short prose "
    "or bullet points for descriptions, features, and applications.\n"
    "- Keep a balanced voice: lead with the technical facts, then, where the context "
    "supports it, note the key selling point or where the product fits. Do not oversell "
    "or claim anything not in the context.\n"
    "- If the answer is not in the catalog, say so plainly and suggest a next step — a "
    "related product from the context, or contacting the supplier for details. Do not guess."
)

_BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)


def load_model(model_id: str) -> tuple:
    logger.info(f"loading {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=_BNB_CONFIG,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def build_prompt(query: str, context_chunks: list[str], max_context_chars: int = 2800) -> str:
    # trim each chunk proportionally so total context stays within budget
    budget = max_context_chars // max(len(context_chunks), 1)
    trimmed = [c[:budget] for c in context_chunks]
    context = "\n\n---\n\n".join(trimmed)
    return f"Catalog context:\n{context}\n\nCustomer question: {query}\n\nAnswer:"


def generate(
    query: str,
    context_chunks: list[str],
    model,
    tokenizer,
    model_id: str,
    max_new_tokens: int = 512,
) -> str:
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": build_prompt(query, context_chunks)},
    ]

    kwargs: dict = {"tokenize": False, "add_generation_prompt": True}
    if "qwen3" in model_id.lower():
        kwargs["enable_thinking"] = False

    prompt_text = tokenizer.apply_chat_template(messages, **kwargs)
    inputs = tokenizer(
        prompt_text, return_tensors="pt", truncation=True, max_length=2048
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()

    if "deepseek-r1" in model_id.lower():
        text = _THINK_RE.sub("", text).strip()

    text = _FOLLOWUP_RE.split(text)[0].strip()
    return text


def generate_raw(prompt_text: str, model, tokenizer, max_new_tokens: int = 128) -> str:
    inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=256).to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


def delete_model_cache(hf_home: str | Path, model_id: str | None = None) -> None:
    hub_dir = Path(hf_home) / "hub"
    if not hub_dir.exists():
        return
    if model_id:
        cache_name = "models--" + model_id.replace("/", "--")
        model_cache_dir = hub_dir / cache_name
        if model_cache_dir.exists():
            shutil.rmtree(model_cache_dir)
            logger.info(f"cleared cache: {model_cache_dir}")
    else:
        shutil.rmtree(hub_dir)
        logger.info(f"cleared cache: {hub_dir}")
