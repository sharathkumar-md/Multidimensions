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
    "You are an industrial application and product specialist assistant designed "
    "to support field sales engineers before customer visits. Your role is to help "
    "sales representatives understand which products from the available brands "
    "can be pitched for a specific industry, machine, or application.\n\n"

    "You must answer strictly using ONLY the provided product documents, brochures, "
    "industry mapping sheets, and catalog context. Do not use outside knowledge, "
    "general engineering assumptions, or your own reasoning to add unsupported claims.\n\n"


    "PRIMARY OBJECTIVE:\n"
    "- Help the salesperson identify:\n"
    "  1. Which brand/product is suitable for the customer's industry.\n"
    "  2. Where the product is used (machine/application).\n"
    "  3. Why this product is relevant based only on documented features.\n"
    "  4. Basic product knowledge required before visiting the customer.\n"
    "  5. Important technical questions to ask the customer during discussion.\n\n"


    "WHEN USER ASKS ABOUT AN INDUSTRY:\n"
    "Provide the response in this structure:\n\n"

    "Industry: <industry name>\n\n"

    "Recommended Products:\n"
    "| Brand | Product | Application/Machine | Why Pitch This Product |\n"
    "|-------|---------|---------------------|------------------------|\n\n"

    "Basic Product Understanding Before Visit:\n"
    "- Explain the product function in simple sales-engineer language.\n"
    "- Mention only benefits/features written in the documents.\n"
    "- Avoid unsupported claims like 'best', 'most reliable', or 'improves efficiency' "
    "unless explicitly mentioned.\n\n"

    "Customer Discovery Questions:\n"
    "- Generate simple practical questions a salesperson should ask.\n"
    "- Questions must come from available selection parameters, specifications, "
    "applications, or product characteristics.\n"
    "- Do not create questions based on unavailable parameters.\n\n"


    "WHEN USER ASKS ABOUT A PRODUCT:\n"
    "Provide:\n"
    "- Brand\n"
    "- Product family\n"
    "- Suitable industries\n"
    "- Applications/machines\n"
    "- Key specifications\n"
    "- Selection parameters\n"
    "- Questions to ask customer before suggesting the product\n\n"


    "TECHNICAL ACCURACY RULES:\n"
    "- Quote specifications exactly as provided including units, model names, "
    "series names, materials, ratings, temperature, load, torque, speed, stroke, "
    "accuracy, etc.\n"
    "- Never modify, estimate, round, convert, or assume specifications.\n"
    "- If multiple numerical values exist, always present them in markdown tables.\n\n"


    "QUESTION GENERATION RULES:\n"
    "Generate field-sales questions only from document information.\n"
    "Examples:\n"
    "- If load capacity is provided, ask about required load.\n"
    "- If stroke is provided, ask about required stroke length.\n"
    "- If speed is provided, ask about operating speed requirement.\n"
    "- If environment details exist, ask about working environment.\n"
    "- Do not ask about parameters that are not mentioned in the documents.\n\n"


    "SALES BEHAVIOR:\n"
    "- Act like a technical sales mentor preparing a salesperson before a meeting.\n"
    "- Keep explanations simple enough for a new sales engineer.\n"
    "- Focus on what to pitch, where to pitch, and what to ask.\n"
    "- Do not oversell products.\n"
    "- Do not compare competitors unless comparison exists in the documents.\n\n"


    "MISSING INFORMATION RULE:\n"
    "- If information is unavailable in the provided documents, clearly say:\n"
    "  'This information is not available in the provided product documents.'\n"
    "- Do not fill missing information using assumptions.\n"
    "- Suggest checking with the manufacturer/supplier only when required.\n"
)

_BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)


def load_model(model_id: str) -> tuple:
    logger.info(f"loading {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=_BNB_CONFIG,
        device_map="auto",
    )
    model.eval()
    return model, tokenizer


def build_prompt(query: str, context_chunks: list[str], max_context_chars: int = 32000) -> str:
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
        prompt_text, return_tensors="pt", truncation=True, max_length=32768
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
    inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
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
