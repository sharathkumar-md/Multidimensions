import json
import re
import sys
import time
from pathlib import Path

from loguru import logger
from PIL import Image

REPO_DIR = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(REPO_DIR / "01.1_ocr_vlm"))
from src.vlm_extractor import load_model, _MAX_PIXELS
from config.settings import settings as vlm_settings

import torch


def parse_markdown_pages(md_path: Path) -> dict[str, str]:
    """Parse the markdown file into a dict of {page_num_str: page_text}."""
    if not md_path.exists():
        return {}
    
    text = md_path.read_text(encoding="utf-8")
    # Split by "## Page X"
    pages = {}
    
    parts = re.split(r"^##\s+Page\s+(\d+)", text, flags=re.MULTILINE)
    # parts[0] is preamble (before any ## Page)
    # parts[1] is page num, parts[2] is content, parts[3] is page num...
    
    for i in range(1, len(parts), 2):
        page_num = parts[i]
        content = parts[i+1].strip()
        pages[page_num] = content
        
    return pages


def run_captioning():
    logger.remove()
    logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}", colorize=True)
    
    fig_index_path = REPO_DIR / "04_demo" / "figure_index.json"
    captions_path = REPO_DIR / "04_demo" / "figure_captions.json"
    
    if not fig_index_path.exists():
        logger.error(f"Figure index not found at {fig_index_path}")
        return
        
    fig_index = json.loads(fig_index_path.read_text(encoding="utf-8"))
    
    existing_captions = {}
    if captions_path.exists():
        try:
            existing_captions = json.loads(captions_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    # Collect tasks
    tasks = []
    for doc_name, doc_data in fig_index.items():
        doc_id = doc_data.get("doc_id")
        figures_base = REPO_DIR / Path(doc_data.get("figures_base", ""))
        md_path = REPO_DIR / "data" / "ocr_output_vlm" / "markdown" / f"{doc_id}.md"
        
        pages_dict = parse_markdown_pages(md_path)
        
        for page_num, figs in doc_data.get("by_page", {}).items():
            page_text = pages_dict.get(page_num, "")
            # Truncate page text if it's absurdly long
            page_text = page_text[:2000]
            
            for fig in figs:
                filename = fig["filename"]
                if filename in existing_captions:
                    continue # Skip already processed
                    
                full_path = figures_base / filename
                if not full_path.exists():
                    continue
                    
                tasks.append({
                    "doc_name": doc_name,
                    "filename": filename,
                    "image_path": full_path,
                    "page_text": page_text
                })

    if not tasks:
        logger.info("All figures already captioned or missing.")
        return

    logger.info(f"Loading VLM to caption {len(tasks)} images...")
    model, processor = load_model(vlm_settings.model_id)
    
    from qwen_vl_utils import process_vision_info

    new_captions_count = 0
    t0 = time.time()
    
    for i, task in enumerate(tasks):
        logger.info(f"Captioning {i+1}/{len(tasks)}: {task['filename']}")
        
        try:
            image = Image.open(task["image_path"]).convert("RGB")
        except Exception as e:
            logger.error(f"Failed to open image {task['filename']}: {e}")
            continue

        prompt = (
            "You are an expert mechanical engineer analyzing an industrial catalog.\n"
            "Write a concise, highly accurate, and searchable caption (1-2 sentences) for this specific image.\n"
            "If the image contains printed part numbers or labels, include them.\n\n"
            "Here is the text extracted from the surrounding PDF page for context:\n"
            "---\n"
            f"{task['page_text']}\n"
            "---\n\n"
            "Caption:"
        )

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image, "max_pixels": _MAX_PIXELS},
                {"type": "text", "text": prompt},
            ],
        }]

        try:
            text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            
            inputs = processor(
                text=[text_prompt],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(model.device)

            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=150,
                    do_sample=False,
                    pad_token_id=processor.tokenizer.eos_token_id,
                )

            new_ids = [out[j][len(inputs.input_ids[j]):] for j in range(len(out))]
            caption = processor.batch_decode(new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0].strip()
            
            existing_captions[task["filename"]] = caption
            new_captions_count += 1
            
            # Save every 10 images to avoid data loss
            if new_captions_count % 10 == 0:
                captions_path.write_text(json.dumps(existing_captions, indent=2), encoding="utf-8")
                
        except Exception as e:
            logger.error(f"Failed to caption {task['filename']}: {e}")
            
        finally:
            # Cleanup
            if 'inputs' in locals(): del inputs
            if 'out' in locals(): del out
            if 'new_ids' in locals(): del new_ids
            torch.cuda.empty_cache()

    # Final save
    captions_path.write_text(json.dumps(existing_captions, indent=2), encoding="utf-8")
    
    elapsed = time.time() - t0
    logger.info(f"Finished captioning {new_captions_count} images in {elapsed:.1f}s.")
    
    # Cleanup model
    del model, processor
    torch.cuda.empty_cache()


if __name__ == "__main__":
    run_captioning()
