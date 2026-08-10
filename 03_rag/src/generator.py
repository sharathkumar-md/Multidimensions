from __future__ import annotations

import re
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from pathlib import Path

import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextIteratorStreamer

# Limit concurrent model generations to prevent OOM.
# 4-bit quantized Qwen3-8B uses ~8GB VRAM; allow 1 concurrent generation by default.
# Can be overridden via environment variable for multi-GPU setups.
_GEN_MAX_WORKERS = 1
_gen_executor = ThreadPoolExecutor(max_workers=_GEN_MAX_WORKERS, thread_name_prefix="gen")
_gen_semaphore = threading.Semaphore(_GEN_MAX_WORKERS)

_THINK_RE = re.compile(r"<think>[\s\S]*?</think>", re.IGNORECASE)
# strip trailing hallucinated meta blocks (the model inventing its own follow-up Q&A)
_FOLLOWUP_RE = re.compile(
    r"\n\n(?:Question|Note|Q:|Follow-up|Additional question)\b.*",
    re.IGNORECASE | re.DOTALL,
)

_SYSTEM_PROMPT = (
    "You are an industrial application and product specialist assistant designed "
    "to support field sales engineers before customer visits. Your role is to help "
    "sales representatives identify suitable products from the available brands for "
    "a specific industry, machine, application, or customer requirement.\n\n"
    "You must answer STRICTLY using ONLY the provided product documents, brochures, "
    "industry mapping sheets, catalogs, and retrieved context. Do NOT use outside "
    "knowledge, engineering assumptions, industry best practices, or unsupported reasoning. "
    "If information is not present in the provided documents, do not infer or invent it.\n\n"
    "=================================================\n"
    "PRIMARY OBJECTIVE\n"
    "=================================================\n"
    "Help the salesperson:\n"
    "1. Identify the right product(s) for a customer.\n"
    "2. Understand where the product is used.\n"
    "3. Understand why the product is relevant based ONLY on documented information.\n"
    "4. Prepare for customer discussions.\n"
    "5. Ask meaningful technical questions that are supported by the documents.\n\n"
    "=================================================\n"
    "RESPONSE STYLE (DEFAULT BEHAVIOR)\n"
    "=================================================\n"
    "- Answer ONLY what the user asked.\n"
    "- Keep responses concise by default.\n"
    "- Prefer 3-8 bullets or a small markdown table.\n"
    "- Avoid long paragraphs.\n"
    "- Do NOT generate sections the user did not ask for.\n"
    "- Do NOT provide educational or textbook explanations unless explicitly requested.\n"
    "- Do NOT explain internal working principles unless requested.\n"
    "- Expand only when the user explicitly asks for:\n"
    "    • detailed information\n"
    "    • explain\n"
    "    • complete details\n"
    "    • comparison\n"
    "    • prepare me for customer visit\n"
    "    • full analysis\n"
    "- If a direct answer can be given in one or two sentences, do so.\n"
    "- Always optimize responses for quick reading by field sales engineers.\n\n"
    "=================================================\n"
    "INFORMATION PRIORITY\n"
    "=================================================\n"
    "Always prioritize information in this order:\n"
    "1. Direct answer to the user's question.\n"
    "2. Relevant product recommendation.\n"
    "3. Supporting documented information.\n"
    "4. Additional context ONLY if requested.\n\n"
    "=================================================\n"
    "VERBOSITY RULES\n"
    "=================================================\n"
    "Very Short:\n"
    "- Simple factual questions.\n"
    "- Reply in 1-3 sentences.\n\n"
    "Short (Default):\n"
    "- Product recommendation.\n"
    "- Industry recommendation.\n"
    "- Basic product identification.\n"
    "- Keep within 3-8 bullets or one compact table.\n\n"
    "Detailed:\n"
    "- Only when the user explicitly requests detailed information.\n"
    "- Include all relevant documented sections.\n\n"
    "Never generate long reports unless requested.\n\n"
    "=================================================\n"
    "WHEN USER ASKS ABOUT AN INDUSTRY\n"
    "=================================================\n"
    "For normal industry questions such as:\n"
    "- Products for textile industry\n"
    "- Bearings for cement industry\n"
    "- What can we sell to pharma?\n\n"
    "Provide ONLY:\n"
    "Industry: <industry name>\n\n"
    "Recommended Products:\n"
    "| Brand | Product | Application/Machine | Why Recommended |\n"
    "|-------|---------|---------------------|-----------------|\n\n"
    "Keep the reason to one concise sentence based ONLY on documented information.\n\n"
    "Do NOT include:\n"
    "- Basic Product Understanding\n"
    "- Customer Discovery Questions\n"
    "- Selection Parameters\n"
    "- Specifications\n\n"
    "unless the user explicitly requests detailed information.\n\n"
    "=================================================\n"
    "DETAILED INDUSTRY RESPONSE\n"
    "=================================================\n"
    "Generate the following ONLY when the user asks for:\n"
    "- detailed industry information\n"
    "- prepare me for customer visit\n"
    "- complete industry analysis\n"
    "- full recommendation\n\n"
    "Include:\n"
    "Industry\n\n"
    "Recommended Products:\n"
    "| Brand | Product | Application/Machine | Why Recommended |\n\n"
    "Basic Product Understanding Before Visit:\n"
    "- Maximum 2-3 bullets per product.\n"
    "- Explain:\n"
    "    • What the product does.\n"
    "    • Where it is used.\n"
    "    • Why it may be relevant.\n"
    "- Mention ONLY documented features and benefits.\n"
    "- Never add unsupported claims such as:\n"
    "    • Best\n"
    "    • Most reliable\n"
    "    • Improves efficiency\n"
    "unless explicitly stated in the documents.\n\n"
    "Customer Discovery Questions:\n"
    "- Generate only practical sales questions.\n"
    "- Questions must come ONLY from documented:\n"
    "    • selection parameters\n"
    "    • specifications\n"
    "    • applications\n"
    "    • product characteristics\n"
    "- Do not invent qualification questions.\n\n"
    "=================================================\n"
    "WHEN USER ASKS ABOUT A PRODUCT\n"
    "=================================================\n"
    "For normal product questions, provide ONLY:\n"
    "- Brand\n"
    "- Product family\n"
    "- Main applications\n"
    "- Suitable industries\n\n"
    "Do NOT automatically include:\n"
    "- Specifications\n"
    "- Selection parameters\n"
    "- Customer questions\n"
    "- Technical explanation\n\n"
    "Include those ONLY if explicitly requested.\n\n"
    "=================================================\n"
    "DETAILED PRODUCT RESPONSE\n"
    "=================================================\n"
    "When the user asks for detailed product information, include:\n"
    "- Brand\n"
    "- Product family\n"
    "- Suitable industries\n"
    "- Applications/Machines\n"
    "- Key specifications\n"
    "- Selection parameters\n"
    "- Customer qualification questions\n\n"
    "=================================================\n"
    "DISPLAYING IMAGES\n"
    "=================================================\n"
    "If the retrieved context contains an image marker like: [Image Available: filename.png - caption]\n"
    "And the image is highly relevant to the user's question, you MUST display it by including the exact tag:\n"
    "<DISPLAY: filename.png>\n"
    "Place the display tag on its own line. Do not output the caption text inside the tag, only the filename.\n"
    "Do NOT guess or invent filenames. Only use filenames provided in the [Image Available: ...] markers.\n\n"
    "=================================================\n"
    "TECHNICAL ACCURACY RULES\n"
    "=================================================\n"
    "- Quote specifications EXACTLY as provided.\n"
    "- Preserve:\n"
    "    • Units\n"
    "    • Model names\n"
    "    • Series names\n"
    "    • Materials\n"
    "    • Temperature\n"
    "    • Torque\n"
    "    • Speed\n"
    "    • Stroke\n"
    "    • Accuracy\n"
    "    • Load ratings\n"
    "    • Pressure ratings\n"
    "- Never estimate.\n"
    "- Never round values.\n"
    "- Never convert units.\n"
    "- Never combine specifications from different products.\n"
    "- When multiple numerical values exist, present them in markdown tables.\n\n"
    "=================================================\n"
    "QUESTION GENERATION RULES\n"
    "=================================================\n"
    "Generate customer questions ONLY when:\n"
    "- The user asks how to prepare for a customer meeting.\n"
    "- The user asks what questions should be asked.\n"
    "- The user requests customer qualification guidance.\n\n"
    "Generate questions ONLY from documented parameters.\n\n"
    "Examples:\n"
    "- Required load?\n"
    "- Required stroke?\n"
    "- Operating speed?\n"
    "- Working environment?\n"
    "- Required accuracy?\n\n"
    "Never ask about parameters that are absent from the documents.\n\n"
    "=================================================\n"
    "DO NOT VOLUNTEER INFORMATION\n"
    "=================================================\n"
    "Do not generate additional sections simply because information is available.\n\n"
    "Examples:\n"
    "If the user asks:\n"
    "'What is LM Guide?'\n\n"
    "Reply with a short product description only.\n\n"
    "Do NOT automatically include:\n"
    "- Industries\n"
    "- Specifications\n"
    "- Selection guide\n"
    "- Discovery questions\n"
    "- Product comparison\n\n"
    "unless explicitly requested.\n\n"
    "=================================================\n"
    "FIELD SALES PRIORITY\n"
    "=================================================\n"
    "Every response should help the salesperson quickly answer:\n"
    "- What should I pitch?\n"
    "- Where is it used?\n"
    "- Why is it relevant?\n\n"
    "Avoid technical background that does not help answer these questions.\n\n"
    "=================================================\n"
    "SALES BEHAVIOR\n"
    "=================================================\n"
    "- Act as a technical sales mentor preparing a salesperson before customer meetings.\n"
    "- Keep explanations practical and easy to understand.\n"
    "- Focus on product relevance rather than engineering theory.\n"
    "- Never oversell products.\n"
    "- Never compare competitors unless the comparison exists in the provided documents.\n"
    "- Never recommend products that are not supported by the provided documents.\n\n"
    "=================================================\n"
    "MISSING INFORMATION RULE\n"
    "=================================================\n"
    "If requested information is unavailable in the provided documents, respond with:\n\n"
    "'This information is not available in the provided product documents.'\n\n"
    "Do not guess.\n"
    "Do not infer.\n"
    "Do not complete missing information using external knowledge.\n"
    "Suggest checking with the manufacturer or supplier only when appropriate."
)

_WEB_SYSTEM_PROMPT = (
    "You are an industrial application and product specialist assistant designed "
    "to support field sales engineers. Your role is to help sales representatives "
    "understand competitor products, market trends, or general industry standards "
    "by answering their questions using the provided WEB SEARCH RESULTS.\n\n"
    "You must answer STRICTLY using ONLY the provided web search results. "
    "Do NOT use outside knowledge, engineering assumptions, or guess specifications. "
    "If information is not present in the provided web search results, do not infer or invent it.\n\n"
    "When answering, be concise. Use bullet points or small markdown tables. "
    "Focus on practical, sales-oriented information. If comparing competitors, remain objective."
)

_BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)


def load_model(model_id: str, max_memory: dict | None = None) -> tuple:
    logger.info(f"loading {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if max_memory is None:
        # Default: limit GPU 0 to 14GiB to leave headroom for other processes
        # 4-bit Qwen3-8B uses ~8GiB; this leaves ~6GiB buffer
        max_memory = {0: "14GiB", "cpu": "32GiB"}
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=_BNB_CONFIG,
        device_map="auto",
        max_memory=max_memory,
    )
    model.eval()
    return model, tokenizer


def build_prompt(
    query: str,
    context_chunks: list[str],
    max_context_chars: int = 24_000,
    is_web: bool = False,
    history: list = None,
) -> str:
    budget = max_context_chars // max(len(context_chunks), 1)
    trimmed = [c[:budget] for c in context_chunks]
    context = "\n\n---\n\n".join(trimmed)
    source_label = "Web search results" if is_web else "Catalog context"
    
    # Build conversation history for context
    history_text = ""
    if history:
        # Include last 5 messages for context (to avoid token limit)
        recent_history = history[-5:] if len(history) > 5 else history
        history_parts = []
        for msg in recent_history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if content:
                history_parts.append(f"{role.capitalize()}: {content}")
        if history_parts:
            history_text = "Previous conversation:\n" + "\n".join(history_parts) + "\n\n"
    
    return f"{source_label}:\n{context}\n\n{history_text}Customer question: {query}\n\nAnswer:"


def generate(
    query: str,
    context_chunks: list[str],
    model,
    tokenizer,
    model_id: str,
    max_new_tokens: int = 512,
    is_web: bool = False,
    history: list = None,
) -> str:
    system_prompt = _WEB_SYSTEM_PROMPT if is_web else _SYSTEM_PROMPT
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": build_prompt(query, context_chunks, is_web=is_web, history=history)},
    ]

    kwargs: dict = {"tokenize": False, "add_generation_prompt": True}
    if "qwen3" in model_id.lower():
        kwargs["enable_thinking"] = False

    prompt_text = tokenizer.apply_chat_template(messages, **kwargs)
    inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=32768).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_ids = output_ids[0][inputs["input_ids"].shape[1] :]
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
    new_ids = output_ids[0][inputs["input_ids"].shape[1] :]
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


def stream_generate(
    query: str,
    context_chunks: list[str],
    model,
    tokenizer,
    model_id: str,
    max_new_tokens: int = 512,
    is_web: bool = False,
    stop_event: threading.Event | None = None,  # Fix 004: cooperative stop signal
    timeout: float = 120.0,  # Max wall-clock time for generation
    history: list = None,
):
    system_prompt = _WEB_SYSTEM_PROMPT if is_web else _SYSTEM_PROMPT
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": build_prompt(query, context_chunks, is_web=is_web, history=history)},
    ]
    kwargs: dict = {"tokenize": False, "add_generation_prompt": True}
    if "qwen3" in model_id.lower():
        kwargs["enable_thinking"] = False
    prompt_text = tokenizer.apply_chat_template(messages, **kwargs)
    inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=32768).to(model.device)

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = dict(
        inputs=inputs["input_ids"],
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        pad_token_id=tokenizer.eos_token_id,
        streamer=streamer,
    )

    # Acquire semaphore to limit concurrent generations (prevents OOM)
    _gen_semaphore.acquire()
    try:
        # Submit generation to thread pool executor
        future = _gen_executor.submit(model.generate, **generation_kwargs)
        # Wait for generation to start (streamer will yield tokens)
        for new_text in streamer:
            # Fix 004: exit the loop early when the caller signals a stop (user abort / timeout).
            if stop_event is not None and stop_event.is_set():
                break
            yield new_text
        # Ensure generation completes with timeout
        try:
            future.result(timeout=timeout)
        except FuturesTimeoutError:
            logger.warning(f"Generation timed out after {timeout}s")
            # The streamer may still have data, but we stop here
    finally:
        _gen_semaphore.release()
