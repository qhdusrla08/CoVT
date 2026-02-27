import os
import re
import json
import csv
import time
import base64
import random
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List, Iterable

from tqdm import tqdm
from PIL import Image

from openai import OpenAI

# =========================
# Config
# =========================
OUTPUT_DIR = "outputs"
EVAL_DIR = "eval_outputs"
os.makedirs(EVAL_DIR, exist_ok=True)

# NEW: inputs
ORIGINAL_TXT_DIR = os.path.join(OUTPUT_DIR, "original_txt")

# covt input can be a file (jsonl) OR a directory containing json/jsonl files
COVT_INPUT_PATH = os.getenv("COVT_INPUT", os.path.join(OUTPUT_DIR, "captions_covt_P3.jsonl"))
COVT_PROMPT_ID_FALLBACK = os.getenv("COVT_PROMPT_ID", "P3")

JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gpt-4.1-mini")
LIMIT = int(os.getenv("EVAL_LIMIT", "0"))

RANDOM_SEED = int(os.getenv("EVAL_SEED", "42"))
random.seed(RANDOM_SEED)

MAX_RETRIES = 3
SLEEP_BETWEEN_CALLS_SEC = float(os.getenv("SLEEP_SEC", "0.0"))

PAIR_JSONL = os.path.join(EVAL_DIR, "pairs_ogn_vs_covtP3.jsonl")
EVAL_JSONL = os.path.join(EVAL_DIR, "eval_results_ogn_vs_covtP3.jsonl")
SCORES_CSV = os.path.join(EVAL_DIR, "win_rates_ogn_vs_covtP3.csv")

client = OpenAI()

# =========================
# Caption normalization (kept as-is)
# =========================
def normalize_caption(c: str) -> str:
    if not c:
        return ""
    c = c.replace("<answer>", "").replace("</answer>", "").strip()
    c = c.replace("\n", " ").strip()
    c = re.sub(r"<think>.*?</think>", "", c, flags=re.DOTALL).strip()
    return c

# =========================
# Local image -> base64 data URL (kept as-is)
# =========================
def image_to_data_url(image_path: str, max_side: int = 1024) -> str:
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        nw, nh = int(w * scale), int(h * scale)
        img = img.resize((nw, nh), Image.BICUBIC)

    import io
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"

# =========================
# Helpers: loading inputs
# =========================
def list_files_recursive(root: str, exts: Tuple[str, ...]) -> List[str]:
    out = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith(exts):
                out.append(os.path.join(dirpath, fn))
    return sorted(out)

def read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def load_original_txt(dir_path: str) -> Dict[str, dict]:
    """
    Returns:
      image_id -> {"caption": str, "caption_norm": str, "track": str}
    track: inferred from filename suffix _S/_G/_E if present.
    """
    if not os.path.isdir(dir_path):
        raise RuntimeError(f"original_txt dir not found: {dir_path}")

    mapping: Dict[str, dict] = {}
    txts = list_files_recursive(dir_path, (".txt",))
    if not txts:
        raise RuntimeError(f"No .txt files found under: {dir_path}")

    for p in txts:
        stem = os.path.splitext(os.path.basename(p))[0]  # image_id
        raw = read_text_file(p).strip()

        # 🔧 양 끝 큰따옴표 제거
        if raw.startswith('"') and raw.endswith('"'):
            raw = raw[1:-1].strip()

        cap_norm = normalize_caption(raw)


        # Track = last token after underscore if it's short like S/G/E, else unknown
        track = "unknown"
        parts = stem.split("_")
        if parts:
            last = parts[-1]
            if len(last) <= 2:  # S, G, E, etc.
                track = last

        mapping[stem] = {"caption": raw, "caption_norm": cap_norm, "track": track}
    return mapping

def load_jsonl_rows(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def iter_covt_records(path_or_dir: str) -> Iterable[dict]:
    """
    Supports:
      - a single .jsonl file
      - a directory containing .jsonl and/or .json files (each json can be:
          * a dict with "caption" fields
          * or a list of dicts
        )
    """
    if os.path.isdir(path_or_dir):
        files = list_files_recursive(path_or_dir, (".jsonl", ".json"))
        if not files:
            raise RuntimeError(f"No .json/.jsonl files found under: {path_or_dir}")

        for fp in files:
            if fp.lower().endswith(".jsonl"):
                for r in load_jsonl_rows(fp):
                    yield r
            else:
                with open(fp, "r", encoding="utf-8") as f:
                    obj = json.load(f)
                if isinstance(obj, list):
                    for r in obj:
                        if isinstance(r, dict):
                            yield r
                elif isinstance(obj, dict):
                    yield obj
        return

    # file
    if not os.path.exists(path_or_dir):
        raise RuntimeError(f"covt input not found: {path_or_dir}")

    if path_or_dir.lower().endswith(".jsonl"):
        for r in load_jsonl_rows(path_or_dir):
            yield r
    elif path_or_dir.lower().endswith(".json"):
        with open(path_or_dir, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, list):
            for r in obj:
                if isinstance(r, dict):
                    yield r
        elif isinstance(obj, dict):
            yield obj
    else:
        raise RuntimeError(f"Unsupported covt input extension: {path_or_dir}")

def index_covt_captions(path_or_dir: str) -> Dict[str, dict]:
    """
    Returns:
      image_id -> {"caption": str, "caption_norm": str, "prompt_id": str, "track": str, "image_path": str}
    track is taken from record if exists, else inferred from image_id suffix.
    """
    out: Dict[str, dict] = {}

    for r in iter_covt_records(path_or_dir):
        # If your jsonl has status field, respect it (optional)
        if r.get("status") and r.get("status") != "ok":
            continue

        cap = r.get("caption", "")
        cap_norm = normalize_caption(cap)
        if not cap_norm:
            continue

        image_path = r.get("image_path", "")
        image_id = r.get("image_id")
        if not image_id:
            if image_path:
                image_id = os.path.splitext(os.path.basename(image_path))[0]
            else:
                # last resort: some datasets store filename under "file_name"
                fn = r.get("file_name") or r.get("image") or ""
                image_id = os.path.splitext(os.path.basename(fn))[0] if fn else None

        if not image_id:
            continue

        prompt_id = r.get("prompt_id") or COVT_PROMPT_ID_FALLBACK

        track = r.get("track")
        if not track:
            parts = image_id.split("_")
            track = parts[-1] if parts and len(parts[-1]) <= 2 else "unknown"

        out[image_id] = {
            "caption": cap,
            "caption_norm": cap_norm,
            "prompt_id": prompt_id,
            "track": track,
            "image_path": image_path,
        }

    return out

# =========================
# Build pairs: original_txt vs covt(P3)
# =========================
@dataclass
class PairRecord:
    track: str
    prompt_id: str
    image_id: str
    image_path: str
    original_caption: str
    covt_caption: str

def build_pairs() -> List[PairRecord]:
    orig = load_original_txt(ORIGINAL_TXT_DIR)
    covt = index_covt_captions(COVT_INPUT_PATH)

    orig_keys = set(orig.keys())
    covt_keys = set(covt.keys())

    matched_keys = sorted(orig_keys & covt_keys)
    unmatched_orig = sorted(orig_keys - covt_keys)

    # 🔔 매칭되지 않은 txt 파일명 출력
    if unmatched_orig:
        print(f"[WARN] Unmatched original_txt files ({len(unmatched_orig)}):")
        for k in unmatched_orig:
            print(f"  - {k}")

    if not matched_keys:
        raise RuntimeError(
            "No matched image_ids between original_txt and covt captions.\n"
            f"- original_txt count: {len(orig)}\n"
            f"- covt count: {len(covt)}"
        )

    pairs: List[PairRecord] = []

    for image_id in matched_keys:
        covt_rec = covt[image_id]

        # ✅ Track은 covt jsonl 값만 사용
        track = covt_rec.get("track")
        if not track:
            raise RuntimeError(
                f"Missing 'track' field in covt caption for image_id={image_id}"
            )

        image_path = covt_rec.get("image_path")
        if not image_path:
            raise RuntimeError(
                f"Missing image_path for image_id={image_id} in covt input"
            )

        pairs.append(
            PairRecord(
                track=track,
                prompt_id=covt_rec.get("prompt_id", COVT_PROMPT_ID_FALLBACK),
                image_id=image_id,
                image_path=image_path,
                original_caption=orig[image_id]["caption_norm"],
                covt_caption=covt_rec["caption_norm"],
            )
        )

    return pairs


# =========================
# GPT blind A/B (kept as-is)
# =========================
JUDGE_SYSTEM = (
    "You are an expert evaluator for training data used in image generation and vision-language models.\n\n"
    "Your task is to compare two captions that describe the same image and determine which caption is more suitable and higher quality for training a generative model.\n\n"
    "When making your judgment, strictly prioritize the following criteria:\n\n"
    "1. Visual Grounding:\n"
    "   - The caption should describe only what is directly observable in the image.\n"
    "   - Avoid assumptions, intentions, inferred states, or unseen context.\n\n"
    "2. Neutrality and Objectivity:\n"
    "   - Prefer factual, descriptive language.\n"
    "   - Penalize subjective, emotional, or evaluative expressions (e.g., \"beautiful\", \"majestic\", \"impressive\").\n\n"
    "3. Generalization:\n"
    "   - Prefer captions that do not rely on specific proper nouns, place names, cultural labels, or specialized knowledge unless they are visually undeniable.\n"
    "   - Captions should generalize well across cultures and contexts.\n\n"
    "4. Absence of Hallucination:\n"
    "   - The caption must not include incorrect, unverifiable, or image-inconsistent details (e.g., wrong time of day, lighting, weather, or object identity).\n\n"
    "5. Appropriate Level of Detail:\n"
    "   - Prefer concise yet sufficiently informative captions.\n"
    "   - Penalize overly detailed descriptions of minor actions or speculative relationships.\n\n"
    "You must choose exactly one of the two captions as the better option for generative model training.\n\n"
    "Your output format must be:\n"
    "- \"Return JSON only with: {\\\"winner\\\":\\\"A\\\"|\\\"B\\\",\\\"reason\\\":\\\"...\\\"}\"\n"
    "- Followed by a brief explanation (1-3 sentences) justifying your choice based on the criteria above."
)

def safe_json_load(s: str) -> dict:
    s = (s or "").strip()
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if m:
        s = m.group(0)
    return json.loads(s)

def judge_ab(image_path: str, capA: str, capB: str) -> dict:
    data_url = image_to_data_url(image_path, max_side=1024)

    swapped = False
    shownA, shownB = capA, capB
    if random.random() < 0.5:
        shownA, shownB = capB, capA
        swapped = True

    user_text = (
        "Evaluate captions A and B for the given image.\n"
        "Respond with JSON exactly in this schema:\n"
        "{\n"
        '  "winner": "A" | "B" | "tie",\n'
        '  "reason": string\n'
        "}\n"
        "Captions:\n"
        f"A: {shownA}\n"
        f"B: {shownB}\n"
    )

    resp = client.responses.create(
        model=JUDGE_MODEL,
        input=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {
                "role": "user",
                "content": [
                    {"type": "input_image", "image_url": data_url},
                    {"type": "input_text", "text": user_text},
                ],
            },
        ],
        temperature=0.0,
    )

    out = safe_json_load(resp.output_text)
    winner_shown = out.get("winner", "tie")
    reason = out.get("reason", "")

    if winner_shown not in ("A", "B", "tie"):
        winner_shown = "tie"

    winner_orig = winner_shown
    if swapped:
        if winner_shown == "A":
            winner_orig = "B"
        elif winner_shown == "B":
            winner_orig = "A"

    return {
        "winner": winner_orig,
        "winner_shown": winner_shown,
        "swapped": swapped,
        "reason": reason,
    }

# =========================
# Aggregate (kept as-is, but key names updated)
# =========================
def aggregate_from_eval_jsonl(eval_jsonl_path: str) -> Dict[Tuple[str, str], dict]:
    agg: Dict[Tuple[str, str], dict] = {}
    if not os.path.exists(eval_jsonl_path):
        return agg

    with open(eval_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("status") == "fail":
                continue

            track = r.get("track")
            prompt_id = r.get("prompt_id")
            winner_model = r.get("winner_model")
            if not track or not prompt_id or not winner_model:
                continue

            key = (track, prompt_id)
            if key not in agg:
                agg[key] = {"n": 0, "original_win": 0.0, "covt_win": 0.0, "tie": 0}

            agg[key]["n"] += 1

            if winner_model == "original":
                agg[key]["original_win"] += 1.0
            elif winner_model == "covt":
                agg[key]["covt_win"] += 1.0
            else:
                agg[key]["tie"] += 1
                agg[key]["original_win"] += 0.5
                agg[key]["covt_win"] += 0.5

    return agg

# =========================
# Main
# =========================
def main():
    pairs = build_pairs()
    print(f"[INFO] Total matched pairs: {len(pairs)}")

    with open(PAIR_JSONL, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p.__dict__, ensure_ascii=False) + "\n")
    print(f"[INFO] Wrote pairs -> {PAIR_JSONL}")

    pairs_to_eval = pairs[:LIMIT] if LIMIT > 0 else pairs

    for p in tqdm(pairs_to_eval, desc="GPT A/B eval"):
        # ORIGINAL order fixed: A=original_txt, B=covt
        capA = p.original_caption
        capB = p.covt_caption
        capA_tag = "original"
        capB_tag = "covt"

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                out = judge_ab(p.image_path, capA, capB)

                winner = out.get("winner", "tie")
                winner_shown = out.get("winner_shown", "tie")
                swapped = bool(out.get("swapped", False))
                reason = out.get("reason", "")

                if winner == "tie":
                    winner_model = "tie"
                elif winner == "A":
                    winner_model = capA_tag
                elif winner == "B":
                    winner_model = capB_tag
                else:
                    winner_model = "tie"

                shownA_tag, shownB_tag = capA_tag, capB_tag
                shownA, shownB = capA, capB
                if swapped:
                    shownA_tag, shownB_tag = capB_tag, capA_tag
                    shownA, shownB = capB, capA

                rec = {
                    "track": p.track,
                    "prompt_id": p.prompt_id,
                    "image_id": p.image_id,
                    "image_path": p.image_path,

                    "original_caption": p.original_caption,
                    "covt_caption": p.covt_caption,

                    "capA_tag": capA_tag,
                    "capB_tag": capB_tag,
                    "capA": capA,
                    "capB": capB,

                    "swapped": swapped,
                    "shownA_tag": shownA_tag,
                    "shownB_tag": shownB_tag,
                    "shownA": shownA,
                    "shownB": shownB,

                    "judge_winner": winner,
                    "judge_winner_shown": winner_shown,
                    "winner_model": winner_model,
                    "reason": reason,
                    "judge_model": JUDGE_MODEL,
                    "status": "ok",
                }

                with open(EVAL_JSONL, "a", encoding="utf-8") as wf:
                    wf.write(json.dumps(rec, ensure_ascii=False) + "\n")

                if SLEEP_BETWEEN_CALLS_SEC > 0:
                    time.sleep(SLEEP_BETWEEN_CALLS_SEC)

                break

            except Exception as e:
                if attempt == MAX_RETRIES:
                    rec = {
                        "track": p.track,
                        "prompt_id": p.prompt_id,
                        "image_id": p.image_id,
                        "image_path": p.image_path,
                        "status": "fail",
                        "error": repr(e),
                        "judge_model": JUDGE_MODEL,
                    }
                    with open(EVAL_JSONL, "a", encoding="utf-8") as wf:
                        wf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                else:
                    time.sleep(1.0 * attempt)

    print(f"[INFO] Wrote eval logs -> {EVAL_JSONL}")

    agg = aggregate_from_eval_jsonl(EVAL_JSONL)

    with open(SCORES_CSV, "w", newline="", encoding="utf-8") as cf:
        w = csv.writer(cf)
        w.writerow([
            "track", "prompt",
            "num_samples",
            "original_winrate", "covt_winrate",
            "ties",
        ])
        for (track, prompt_id), v in sorted(agg.items()):
            n = v["n"]
            w.writerow([
                track, prompt_id, n,
                f"{v['original_win'] / n:.6f}" if n else "nan",
                f"{v['covt_win'] / n:.6f}" if n else "nan",
                v["tie"],
            ])

    print(f"[DONE] Wrote win rates -> {SCORES_CSV}")

if __name__ == "__main__":
    main()

