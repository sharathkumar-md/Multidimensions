from __future__ import annotations

from functools import lru_cache

import torch
from loguru import logger
from PIL import Image
from transformers import BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration, AutoProcessor

_EXTRACTION_PROMPT = """\
You are extracting content from a page of an industrial product catalog or brochure.

Rules:
- If the page contains product specifications with min/max parameter ranges, output them as a proper markdown table with columns: | Parameter | Min | Max |
- If the page contains descriptive text, product features, or application lists, output clean prose and bullet points
- If both are present, use tables for specs and prose for descriptions
- Remove repeated page headers/footers, watermarks, page numbers, and company branding that add no information
- Do NOT invent or guess any data not clearly visible on the page
- Output clean markdown only, no commentary"""

_BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)


@lru_cache(maxsize=1)
def _load_model(model_id: str) -> tuple:
    logger.info(f"loading VLM: {model_id}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=_BNB_CONFIG,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    logger.info("VLM loaded")
    return model, processor


def extract_page(
    image: Image.Image,
    model_id: str,
    max_new_tokens: int = 1024,
) -> str:
    from qwen_vl_utils import process_vision_info

    model, processor = _load_model(model_id)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": _EXTRACTION_PROMPT},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=processor.tokenizer.eos_token_id,
        )

    new_ids = [output_ids[i][len(inputs.input_ids[i]):] for i in range(len(output_ids))]
    result = processor.batch_decode(new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return result[0].strip()
