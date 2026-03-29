import os
import json
import time
import argparse
from typing import Dict, Any, List, Optional, Tuple
import re
import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from typing import Optional

# ---------------- Prompts (MVP) ----------------

# 한 문장만 생성
PROMPTS = {
    "P0": (
        "Describe the overall scene structure and main objects in exactly one clear sentence, ensuring high fidelity to the image's layout." 
    ),
    "P1": (
        
        "Identify the primary objects and describe their precise spatial and depth relationships (e.g., front-to-back ordering) in exactly one factual sentence. Use your internal visual reasoning to verify positions."
    ),
    "P2": (
        "Identify the primary objects and describe their precise spatial and depth relationships in **exactly one factual sentence**. Use your internal visual reasoning and perception DINO features to verify the scene structure."
    ),
    "P3": (
        "Describe the scene in the image in **exactly one factual sentence**. Use segmentation, depth map, and perception feature information of the image to verify the scene structure."
    )  
}


IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def load_model_and_processor(model_name: str, ckpt: Optional[str] = None):
    """
    Correct loader for Qwen2.5-VL and CoVT models.
    CoVT uses external visual features (SAM / Depth / DINO) via processor + prompt,
    NOT via additional model parameters. Warning about unused weights is expected.
    """

    src = ckpt if ckpt is not None else model_name
    print(f"[INFO] Loading model from: {src}")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        src,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,   # ⭐ 반드시 필요
    ).eval()

    processor = AutoProcessor.from_pretrained(
        src,
        trust_remote_code=True,
        min_pixels=256 * 28 * 28,
        max_pixels=384 * 28 * 28,   # OOM 방지용
    )

    ip = getattr(processor, "image_processor", None)
    print(
        "[PIXELS]",
        "min_pixels:", getattr(ip, "min_pixels", None),
        "max_pixels:", getattr(ip, "max_pixels", None),
    )

    print(
        "[INFO] NOTE: Warnings about unused SAM / Depth / DINO weights are EXPECTED "
        "for CoVT. Visual reasoning is injected via processor + prompt."
    )

    return model, processor



def extract_caption(text: str) -> str:
    # 1) <answer>...</answer>가 있으면 그 안의 내용만 추출
    m = re.search(r"<answer>(.*?)</answer>", text, flags=re.DOTALL)
    if m:
        text = m.group(1).strip()
    else:
        # 2) <think>...</think> 제거
        # CoVT의 시각적 사고 과정을 제거하여 순수 캡션만 남김
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    # 3) 특수 토큰 및 잔여 태그 제거
    text = re.sub(r"<\|.*?\|>|<answer>|</answer>", "", text)
    text = re.sub(r"\n", " ", text)

    return text.strip()


def resize_if_needed(img: Image.Image, max_side: int = 1024) -> Image.Image:
    w, h = img.size
    if max(w, h) <= max_side:
        return img
    scale = max_side / max(w, h)
    nw, nh = int(w * scale), int(h * scale)
    return img.resize((nw, nh), Image.BICUBIC)


def run_single_inference(
    model,
    processor,
    image_path: str,
    question: str,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    top_p: float = 0.9,
    seed: int = 42,
):
    pil_image = Image.open(image_path).convert("RGB")
    pil_image = resize_if_needed(pil_image, max_side=1024)

    image_ref = image_path

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_ref},
                {"type": "text", "text": question},
            ],
        }
    ]

    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = processor(text=[prompt], images=[pil_image], return_tensors="pt")

    device = model.device
    inputs = {
        k: (v.to(device) if isinstance(v, torch.Tensor) else v)
        for k, v in inputs.items()
    }

    seed = int(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    start = time.time()
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=float(temperature),
            top_p=float(top_p),
            do_sample=(float(temperature) > 0.0),
            pad_token_id=processor.tokenizer.eos_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
        )
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - start

    input_len = inputs["input_ids"].shape[1]
    new_tokens = generated_ids[0, input_len:]
    raw = processor.decode(new_tokens, skip_special_tokens=True)
    answer = extract_caption(raw)

    return answer, elapsed



def append_jsonl(path: str, obj: Dict[str, Any]):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def parse_track_dirs(items: List[str]) -> Dict[str, str]:
    """
    Parse ["A=/path/to/A", "B=/path/to/B", "C=/path/to/C"] -> {"A": "...", ...}
    """
    out = {}
    for it in items:
        if "=" not in it:
            raise ValueError(f"Invalid --track_dirs item: {it} (expected like A=/path)")
        k, v = it.split("=", 1)
        k = k.strip().upper()
        v = v.strip()
        if k not in {"A", "B", "C"}:
            raise ValueError(f"Track must be A/B/C, got: {k}")
        out[k] = v
    return out


def list_images_in_dir(root: str, recursive: bool = True) -> List[str]:
    paths = []
    if recursive:
        for dp, _, files in os.walk(root):
            for fn in files:
                ext = os.path.splitext(fn)[1].lower()
                if ext in IMG_EXTS:
                    paths.append(os.path.join(dp, fn))
    else:
        for fn in os.listdir(root):
            ext = os.path.splitext(fn)[1].lower()
            if ext in IMG_EXTS:
                paths.append(os.path.join(root, fn))
    paths.sort()
    return paths


def load_done_ids(jsonl_path: str) -> set:
    done = set()
    if not os.path.exists(jsonl_path):
        return done
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                if "image_id" in obj:
                    done.add(obj["image_id"])
            except Exception:
                pass
    return done


def make_image_id(path: str) -> str:
    # default: filename without extension
    base = os.path.basename(path)
    return os.path.splitext(base)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track_dirs", nargs="+", required=True,
                    help='Track directories like: A=/path/to/A B=/path/to/B C=/path/to/C (provide what you have)')
    ap.add_argument("--out_dir", default="./outputs", help="Output directory")
    ap.add_argument("--model_name", default="Qwen/Qwen2.5-VL-7B-Instruct", help="HF model repo id")     # Qwen/Qwen2.5-VL-7B-Instruct or Wakals/CoVT-7B-seg_depth_dino
    ap.add_argument("--ckpt", default=None, help="Local checkpoint path (optional)")
    ap.add_argument("--model_tag", default="base", help="Tag for output file naming (e.g., base, covt)")    # base or covt
    ap.add_argument("--prompt_id", default="P1", choices=["P0", "P1", "P2", "P3"])    # P0 or P1 or P2 or P3
    ap.add_argument("--max_new_tokens", type=int, default=96)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--recursive", action="store_true", help="Recursively search images in track dirs")
    ap.add_argument("--skip_existing", action="store_true", help="Skip if image_id already exists in output jsonl")
    ap.add_argument("--limit_per_track", type=int, default=0, help="0 means no limit; otherwise cap each track")
    args = ap.parse_args()

    print("[ARGS]", args)

    track_dirs = parse_track_dirs(args.track_dirs)
    os.makedirs(args.out_dir, exist_ok=True)

    out_path = os.path.join(args.out_dir, f"captions_{args.model_tag}_{args.prompt_id}.jsonl")
    prompt = PROMPTS[args.prompt_id]

    done = load_done_ids(out_path) if args.skip_existing else set()

    # Collect image list from dirs
    items: List[Tuple[str, str, str]] = []  # (track, image_id, image_path)
    for track, d in sorted(track_dirs.items()):
        if not os.path.isdir(d):
            raise FileNotFoundError(f"Track dir not found: {track} -> {d}")

        paths = list_images_in_dir(d, recursive=args.recursive)
        if args.limit_per_track and args.limit_per_track > 0:
            paths = paths[: args.limit_per_track]

        for p in paths:
            image_id = make_image_id(p)
            items.append((track, image_id, p))

    print(f"[INFO] Total images collected: {len(items)}")
    for t in ["A", "B", "C"]:
        n = sum(1 for x in items if x[0] == t)
        if t in track_dirs:
            print(f"[INFO] Track {t}: {n} images from {track_dirs[t]}")

    print(f"[INFO] Output: {out_path}")
    print(f"[INFO] Model: {args.model_name} (tag={args.model_tag})")
    print(f"[INFO] Prompt({args.prompt_id}): {prompt}")

    model, processor = load_model_and_processor(args.model_name, args.ckpt)

    n_ok, n_fail, n_skip = 0, 0, 0
    for idx, (track, image_id, image_path) in enumerate(items, 1):
        if args.skip_existing and image_id in done:
            n_skip += 1
            continue

        rec: Dict[str, Any] = {
            "image_id": image_id,
            "image_path": image_path,
            "track": track,
            "model": args.model_tag,
            "prompt_id": args.prompt_id,
            "gen": {
                "temperature": args.temperature,
                "top_p": args.top_p,
                "max_new_tokens": args.max_new_tokens,
                "seed": args.seed,
            },
        }

        try:
            caption, elapsed = run_single_inference(
                model=model,
                processor=processor,
                image_path=image_path,
                question=prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                seed=args.seed,
            )
            rec["caption"] = caption
            rec["elapsed_sec"] = elapsed
            rec["status"] = "ok"
            n_ok += 1
        except Exception as e:
            rec["caption"] = ""
            rec["elapsed_sec"] = None
            rec["status"] = "fail"
            rec["error"] = repr(e)
            n_fail += 1

        append_jsonl(out_path, rec)

        if idx % 20 == 0:
            print(f"[INFO] {idx}/{len(items)} processed (ok={n_ok}, fail={n_fail}, skip={n_skip})")

    print(f"[DONE] ok={n_ok}, fail={n_fail}, skip={n_skip}")
    print(f"[DONE] saved to: {out_path}")


if __name__ == "__main__":
    main()
