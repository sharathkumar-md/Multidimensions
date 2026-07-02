

import torch
from loguru import logger
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

_PROMPT = """\
Extract the content from this industrial product catalog or brochure page as clean markdown.

- Product specs with parameter ranges → proper table with columns: | Parameter | Min | Max |
- Descriptive text, features, application lists → clean prose and bullet points
- If both exist on the page, use tables for specs and prose for descriptions
- Strip repeated headers, footers, page numbers, and company branding
- Never invent data that isn't visible on the page
- Output markdown only, no explanation"""

_BNB = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)


_MAX_PIXELS = 512 * 28 * 28  


def load_model(model_id: str):
    """Load the Qwen2.5-VL vision-language model.

    BUG-011: always use Qwen2_5_VLForConditionalGeneration (already imported
    at module level) instead of the generic AutoModelForCausalLM, which may
    silently skip the VL image-processing head.
    """
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=_BNB,
        device_map="auto",
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(
        model_id, max_pixels=_MAX_PIXELS
    )
    return model, processor


def extract_page(image: Image.Image, model, processor, max_new_tokens: int = 1024) -> str:
    from qwen_vl_utils import process_vision_info

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image, "max_pixels": _MAX_PIXELS},
            {"type": "text", "text": _PROMPT},
        ],
    }]

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
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=processor.tokenizer.eos_token_id,
        )

    new_ids = [out[i][len(inputs.input_ids[i]):] for i in range(len(out))]
    result = processor.batch_decode(new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0].strip()

    del inputs, out, new_ids
    torch.cuda.empty_cache()

    return result
