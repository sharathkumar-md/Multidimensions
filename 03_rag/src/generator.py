from __future__ import annotations

import re
import shutil
from pathlib import Path

import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

_THINK_RE = re.compile(r"<think>[\s\S]*?</think>", re.IGNORECASE)

_SYSTEM_PROMPT = (
    "You are a technical assistant. Answer using only the provided context. "
    "If the answer isn't there, say so. Be concise."
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
        torch_dtype=torch.bfloat16,
    )
    model.eval()
    return model, tokenizer


def build_prompt(query: str, context_chunks: list[str]) -> str:
    context = "\n\n---\n\n".join(context_chunks)
    return f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"


def generate(
    query: str,
    context_chunks: list[str],
    model,
    tokenizer,
    model_id: str,
    max_new_tokens: int = 256,
) -> str:
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": build_prompt(query, context_chunks)},
    ]

    kwargs: dict = {"tokenize": False, "add_generation_prompt": True}
    if "qwen3" in model_id.lower():
        kwargs["enable_thinking"] = False

    prompt_text = tokenizer.apply_chat_template(messages, **kwargs)
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

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

    return text


def generate_raw(prompt_text: str, model, tokenizer, max_new_tokens: int = 128) -> str:
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
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


def delete_model_cache(hf_home: str | Path) -> None:
    hub_dir = Path(hf_home) / "hub"
    if hub_dir.exists():
        shutil.rmtree(hub_dir)
        logger.info(f"cleared cache: {hub_dir}")
