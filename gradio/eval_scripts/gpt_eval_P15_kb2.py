"""
gpt_eval_P15_kb2.py
  - captions_base_20_P15.jsonl  (base,  640 samples, 20/category)
  - captions_covt_20_P15.jsonl  (covt,  640 samples, 20/category, KB v2/v3)
  위 두 파일을 직접 지정하여 GPT-4.1 pairwise 평가를 수행.

Output (eval_outputs_korean/):
  pairs_P15_kb2_640_4.1.jsonl
  eval_results_P15_kb2_640_4.1.jsonl
  win_rates_P15_kb2_640_4.1.csv
"""
import os
import re
import json
import csv
import time
import base64
import random
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

from tqdm import tqdm
from PIL import Image

from openai import OpenAI

# =========================
# Config
# =========================
OUTPUT_DIR = "outputs_korean"
EVAL_DIR   = "eval_outputs_korean"
os.makedirs(EVAL_DIR, exist_ok=True)

BASE_FILE = os.path.join(OUTPUT_DIR, "captions_base_20_P15.jsonl")
COVT_FILE = os.path.join(OUTPUT_DIR, "captions_covt_20_P15.jsonl")

JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gpt-4.1")
LIMIT       = int(os.getenv("EVAL_LIMIT", "0"))

RANDOM_SEED = int(os.getenv("EVAL_SEED", "42"))
random.seed(RANDOM_SEED)

MAX_RETRIES             = 3
SLEEP_BETWEEN_CALLS_SEC = float(os.getenv("SLEEP_SEC", "0.0"))

PAIR_JSONL  = os.path.join(EVAL_DIR, "pairs_P15_kb2_640_4.1.jsonl")
EVAL_JSONL  = os.path.join(EVAL_DIR, "eval_results_P15_kb2_640_4.1.jsonl")
SCORES_CSV  = os.path.join(EVAL_DIR, "win_rates_P15_kb2_640_4.1.csv")

client = OpenAI()

# =========================
# Caption normalization
# =========================
def normalize_caption(c: str) -> str:
    if not c:
        return ""
    c = c.replace("<answer>", "").replace("</answer>", "").strip()
    c = c.replace("\n", " ").strip()
    c = re.sub(r"<think>.*?</think>", "", c, flags=re.DOTALL).strip()
    return c

# =========================
# Local image -> base64 data URL
# =========================
def image_to_data_url(image_path: str, max_side: int = 1024) -> str:
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.BICUBIC)

    import io
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"

# =========================
# File loading / indexing
# =========================
@dataclass(frozen=True)
class ItemKey:
    track: str
    prompt_id: str
    image_id: str
    image_path: str

@dataclass
class CaptionItem:
    key: ItemKey
    model_tag: str
    caption: str
    caption_norm: str

def load_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def index_captions(jsonl_path: str) -> Dict[ItemKey, CaptionItem]:
    rows = load_jsonl(jsonl_path)
    out: Dict[ItemKey, CaptionItem] = {}
    for r in rows:
        if r.get("status") != "ok":
            continue
        # track 필드 없을 경우 "B" 기본값 (Track B Korean)
        track      = r.get("track") or "B"
        prompt_id  = r.get("prompt_id")
        image_path = r.get("image_path")
        image_id   = r.get("image_id") or os.path.splitext(os.path.basename(image_path))[0]
        if not prompt_id or not image_path:
            continue

        cap      = r.get("caption", "")
        cap_norm = normalize_caption(cap)
        if not cap_norm:
            continue

        # image_path는 절대/상대 모두 허용; 매칭 키는 image_id+prompt_id 기준
        key = ItemKey(track=track, prompt_id=prompt_id, image_id=image_id, image_path=image_path)
        out[key] = CaptionItem(key=key, model_tag=r.get("model", ""), caption=cap, caption_norm=cap_norm)
    return out

# =========================
# Build pairs
# =========================
@dataclass
class PairRecord:
    track: str
    prompt_id: str
    image_id: str
    image_path: str
    base_caption: str
    covt_caption: str

def build_pairs() -> List[PairRecord]:
    print(f"[INFO] Base file : {BASE_FILE}")
    print(f"[INFO] CoVT file : {COVT_FILE}")

    base_idx = index_captions(BASE_FILE)
    covt_idx = index_captions(COVT_FILE)

    # image_path가 절대/상대 경로로 다를 수 있으므로 (prompt_id, image_id)만으로 매칭
    # base 기준 dict: (prompt_id, image_id) -> ItemKey
    base_lookup: Dict[Tuple[str, str], ItemKey] = {
        (k.prompt_id, k.image_id): k for k in base_idx
    }
    covt_lookup: Dict[Tuple[str, str], ItemKey] = {
        (k.prompt_id, k.image_id): k for k in covt_idx
    }

    common = sorted(set(base_lookup.keys()) & set(covt_lookup.keys()))
    pairs = []
    for pid, iid in common:
        bk = base_lookup[(pid, iid)]
        ck = covt_lookup[(pid, iid)]
        pairs.append(PairRecord(
            track=bk.track,
            prompt_id=bk.prompt_id,
            image_id=bk.image_id,
            image_path=bk.image_path,   # base 파일의 image_path 사용
            base_caption=base_idx[bk].caption_norm,
            covt_caption=covt_idx[ck].caption_norm,
        ))
    return pairs

# =========================
# GPT judge
# =========================
JUDGE_SYSTEM = (
    "You are an expert evaluator for captions intended to fine-tune a multimodal generative model with Korean cultural and domain knowledge (B_korean track).\n\n"
    "Your task is to compare two captions describing the same image and choose which caption is more suitable as fine-tuning training data for Korean-knowledge injection.\n\n"
    "IMPORTANT GOAL:\n"
    "- The best caption should help the model learn Korean cultural/domain concepts AND their correct visual cues.\n"
    "- It must remain grounded in what is visually supported by the image. Do not reward hallucinations.\n"
    "- However, using correct Korean cultural terms for visually present objects is NOT hallucination — it is the core value of this task.\n\n"
    "Judge by the criteria below. Criteria 1 and 2 carry EQUAL weight; evaluate them together rather than in strict cascade:\n\n"
    "1. Visual Grounding & Verifiability:\n"
    "   - The caption must describe what is directly observable in the image.\n"
    "   - Penalize fabricated details (wrong counts, non-existent objects, invented specifics) — these are true hallucinations.\n"
    "   - IMPORTANT: Naming a Korean cultural item (e.g., Kimjang, Jesa, Jeontonghonrye, hanbok) that is visually present is NOT overreach. "
    "It is correct cultural identification and should be rewarded, not penalized.\n"
    "   - Only penalize unverifiable claims that go beyond cultural identification (specific dates, personal identities, brand names, historical event details not visible in the image).\n\n"
    "2. Korean Knowledge Injection Value (EQUAL priority with Criterion 1):\n"
    "   - Prefer captions that explicitly name Korean cultural items, architecture, food, symbols, clothing, rituals, holidays, or domain-specific objects WHEN visually justified.\n"
    "   - Prefer captions that connect the term to distinguishing visual attributes (e.g., garments, patterns, materials, structures, typical components).\n"
    "   - Reward correct Korean-specific taxonomy over generic wording (e.g., 'hanbok' over 'traditional dress', 'Japchae' over 'stir-fried noodles', 'ikseongwan' over 'winged crown').\n"
    "   - PENALIZE generic wording when a specific Korean term is clearly applicable and visually supported.\n\n"
    "3. Disambiguation & Specificity Without Overreach:\n"
    "   - Prefer captions that reduce ambiguity using visually grounded cues.\n"
    "   - Do NOT add non-visual backstory. Use 'appears to be' ONLY when needed and still visually supported.\n\n"
    "4. Cultural Accuracy & Terminology Quality:\n"
    "   - Prefer correct, standard Korean cultural terms and accurate usage.\n"
    "   - Penalize incorrect or mismatched terms (e.g., confusing Japanese/Chinese items with Korean ones, or mislabeling).\n\n"
    "5. Training Usefulness (format & density):\n"
    "   - Prefer a single coherent caption that is concise but information-dense.\n"
    "   - Avoid flowery or emotional language; mild neutral descriptors are fine.\n"
    "   - Penalize overly long lists of minor details that do not help recognition of Korean concepts.\n\n"
    "DECISION RULES:\n"
    "- If one caption hallucinates (fabricates non-existent objects or incorrect details), the other caption should win.\n"
    "- Evaluate Criteria 1 and 2 together: a caption with slightly fewer visual details but correct Korean terminology may be EQUAL or BETTER than one with more visual details but only generic terms.\n"
    "- Use Criteria 3→4→5 as tie-breakers when Criteria 1+2 produce no clear winner.\n"
    "- MICRO TIE-BREAKERS (apply when all criteria produce no clear winner):\n"
    "  (a) Prefer the caption that uses more accurate Korean cultural terminology.\n"
    "  (b) Prefer the caption that describes more observable visual details (colors, shapes, spatial arrangement).\n"
    "  (c) Prefer the caption that is more concise and training-efficient.\n"
    "- TIE POLICY: Declare a tie when both captions are comparable in visual grounding AND Korean knowledge injection value, even if minor wording differences exist. "
    "A tie is appropriate when the differences are too small to meaningfully affect fine-tuning quality.\n\n"
    "Your output format must be:\n"
    "Return JSON only with: {\"winner\":\"A\"|\"B\"|\"tie\",\"reason\":\"...\"}\n"
    "The reason must be 1-3 sentences, explicitly referencing the criteria above (visual grounding + Korean injection value + accuracy)."
)


def safe_json_load(s: str) -> dict:
    s = (s or "").strip()
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if m:
        s = m.group(0)
    return json.loads(s)

def judge_ab(image_path: str, capA: str, capB: str, prompt_id: str = "P15") -> dict:
    data_url = image_to_data_url(image_path, max_side=1024)

    swapped = False
    shownA, shownB = capA, capB
    if random.random() < 0.5:
        shownA, shownB = capB, capA
        swapped = True

    user_text = (
        "Compare captions A and B for the given image.\n"
        "Choose the caption that is more suitable for fine-tuning a model with Korean cultural and domain knowledge.\n\n"
        "Respond with JSON exactly in this schema:\n"
        "{\n"
        '  "winner": "A" | "B" | "tie",\n'
        '  "reason": string\n'
        "}\n\n"
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
# Aggregate win rates
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
                agg[key] = {"n": 0, "base_win": 0, "covt_win": 0, "tie": 0}

            agg[key]["n"] += 1
            if winner_model == "base":
                agg[key]["base_win"] += 1
            elif winner_model == "covt":
                agg[key]["covt_win"] += 1
            else:
                agg[key]["tie"] += 1

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
        capA     = p.base_caption
        capB     = p.covt_caption
        capA_tag = "base"
        capB_tag = "covt"

        if capA.strip() == capB.strip():
            rec = {
                "track": p.track, "prompt_id": p.prompt_id,
                "image_id": p.image_id, "image_path": p.image_path,
                "base_caption": p.base_caption, "covt_caption": p.covt_caption,
                "capA_tag": capA_tag, "capB_tag": capB_tag, "capA": capA, "capB": capB,
                "swapped": False,
                "shownA_tag": capA_tag, "shownB_tag": capB_tag, "shownA": capA, "shownB": capB,
                "judge_winner": "tie", "judge_winner_shown": "tie",
                "winner_model": "tie",
                "reason": "Identical captions: automatic tie.",
                "judge_model": "skip (identical)", "status": "ok",
            }
            with open(EVAL_JSONL, "a", encoding="utf-8") as wf:
                wf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            continue

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                out = judge_ab(p.image_path, capA, capB, prompt_id=p.prompt_id)

                winner       = out.get("winner", "tie")
                winner_shown = out.get("winner_shown", "tie")
                swapped      = bool(out.get("swapped", False))
                reason       = out.get("reason", "")

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
                    "track": p.track, "prompt_id": p.prompt_id,
                    "image_id": p.image_id, "image_path": p.image_path,
                    "base_caption": p.base_caption, "covt_caption": p.covt_caption,
                    "capA_tag": capA_tag, "capB_tag": capB_tag, "capA": capA, "capB": capB,
                    "swapped": swapped,
                    "shownA_tag": shownA_tag, "shownB_tag": shownB_tag,
                    "shownA": shownA, "shownB": shownB,
                    "judge_winner": winner, "judge_winner_shown": winner_shown,
                    "winner_model": winner_model,
                    "reason": reason,
                    "judge_model": JUDGE_MODEL, "status": "ok",
                }

                with open(EVAL_JSONL, "a", encoding="utf-8") as wf:
                    wf.write(json.dumps(rec, ensure_ascii=False) + "\n")

                if SLEEP_BETWEEN_CALLS_SEC > 0:
                    time.sleep(SLEEP_BETWEEN_CALLS_SEC)

                break

            except Exception as e:
                if attempt == MAX_RETRIES:
                    rec = {
                        "track": p.track, "prompt_id": p.prompt_id,
                        "image_id": p.image_id, "image_path": p.image_path,
                        "status": "fail", "error": repr(e), "judge_model": JUDGE_MODEL,
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
            "base_winrate_soft", "covt_winrate_soft",
            "base_winrate_strict", "covt_winrate_strict",
            "ties", "tie_rate",
        ])
        for (track, prompt_id), v in sorted(agg.items()):
            n = v["n"]
            bw, cw, t = v["base_win"], v["covt_win"], v["tie"]
            decisive = bw + cw
            w.writerow([
                track, prompt_id, n,
                f"{(bw + t * 0.5) / n:.6f}" if n else "nan",
                f"{(cw + t * 0.5) / n:.6f}" if n else "nan",
                f"{bw / decisive:.6f}" if decisive else "nan",
                f"{cw / decisive:.6f}" if decisive else "nan",
                t,
                f"{t / n:.6f}" if n else "nan",
            ])

    print(f"[DONE] Wrote win rates -> {SCORES_CSV}")


if __name__ == "__main__":
    main()
