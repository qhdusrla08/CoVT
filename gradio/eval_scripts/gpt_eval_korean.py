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
EVAL_DIR = "eval_outputs_korean"
os.makedirs(EVAL_DIR, exist_ok=True)

JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gpt-4.1")
LIMIT = int(os.getenv("EVAL_LIMIT", "0"))

RANDOM_SEED = int(os.getenv("EVAL_SEED", "42"))
random.seed(RANDOM_SEED)

MAX_RETRIES = 3
SLEEP_BETWEEN_CALLS_SEC = float(os.getenv("SLEEP_SEC", "0.0"))

PAIR_JSONL = os.path.join(EVAL_DIR, "pairs_P15_kb2_4.1.jsonl")
EVAL_JSONL = os.path.join(EVAL_DIR, "eval_results_P15_kb2_4.1.jsonl")
SCORES_CSV = os.path.join(EVAL_DIR, "win_rates_P15_kb2_4.1.csv")

client = OpenAI()

# =========================
# Caption normalization: FIRST SENTENCE ONLY
# =========================
def normalize_caption(c: str) -> str:
    if not c:
        return ""
    c = c.replace("<answer>", "").replace("</answer>", "").strip()
    c = c.replace("\n", " ").strip()
    c = re.sub(r"<think>.*?</think>", "", c, flags=re.DOTALL).strip()

# --- 아래 두 부분을 주석 처리하세요 ---
    
    # (1) 첫 줄만 가져오는 로직
    # c = c.split("\n")[0].strip() 
    
    # (2) 첫 번째 마침표/느낌표/물음표까지만 절단하는 정규표현식 로직
    # m = re.match(r"^(.*?[.!?])(\s|$)", c)
    # if m:
    #     c = m.group(1).strip()
    
    # -----------------------------------

    return c

# =========================
# Local image -> base64 data URL
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
        track = r.get("track")
        prompt_id = r.get("prompt_id")
        image_path = r.get("image_path")
        image_id = r.get("image_id") or os.path.splitext(os.path.basename(image_path))[0]
        if not track or not prompt_id or not image_path:
            continue

        cap = r.get("caption", "")
        cap_norm = normalize_caption(cap)
        if not cap_norm:
            continue

        key = ItemKey(track=track, prompt_id=prompt_id, image_id=image_id, image_path=image_path)
        out[key] = CaptionItem(key=key, model_tag=r.get("model", ""), caption=cap, caption_norm=cap_norm)
    return out

def find_caption_files(output_dir: str) -> List[str]:
    return sorted([f for f in os.listdir(output_dir) if f.endswith(".jsonl")])

def parse_model_prompt_from_filename(fname: str) -> Optional[Tuple[str, str]]:
    m = re.match(r"^captions_([\w]+)_(P\d+)\.jsonl$", fname)
    if not m:
        return None
    return m.group(1), m.group(2)

# =========================
# Build pairs (A): base vs covt per image
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
    files = find_caption_files(OUTPUT_DIR)

    base_files = {}
    covt_files = {}
    for fn in files:
        parsed = parse_model_prompt_from_filename(fn)
        if not parsed:
            continue
        model_tag, prompt_id = parsed
        if model_tag.startswith("base"):
            base_files[prompt_id] = os.path.join(OUTPUT_DIR, fn)
        else:
            covt_files[prompt_id] = os.path.join(OUTPUT_DIR, fn)

    pairs: List[PairRecord] = []
    common_prompts = sorted(set(base_files.keys()) & set(covt_files.keys()))

    # >>> 특정 프롬프트만 평가하려면 아래 줄 활성화, 전체 평가로 복구하려면 다시 주석 처리 <<<
    common_prompts = [p for p in common_prompts if p in ("P15",)]

    if not common_prompts:
        raise RuntimeError("No matching base/covt prompt pairs found in outputs/")

    for pid in common_prompts:
        base_idx = index_captions(base_files[pid])

        covt_idx = index_captions(covt_files[pid])

        keys = sorted(set(base_idx.keys()) & set(covt_idx.keys()), key=lambda k: (k.track, k.image_id))
        for k in keys:
            pairs.append(
                PairRecord(
                    track=k.track,
                    prompt_id=k.prompt_id,
                    image_id=k.image_id,
                    image_path=k.image_path,
                    base_caption=base_idx[k].caption_norm,
                    covt_caption=covt_idx[k].caption_norm,
                )
            )

    return pairs

# =========================
# GPT blind A/B (B_korean) - Knowledge Injection Judge
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
    # 응답이 혹시라도 JSON 외 텍스트를 섞으면, 첫 JSON 블록만 추출
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if m:
        s = m.group(0)
    return json.loads(s)

def judge_ab(image_path: str, capA: str, capB: str, prompt_id: str = "P6") -> dict:
    """
    Input capA/capB is the ORIGINAL order passed by main (we'll keep main fixed: A=base, B=covt).
    Here we RANDOMLY SWAP what we show to the judge, then map winner back to ORIGINAL order.
    """
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

    # Map winner back to ORIGINAL (capA/capB) order
    winner_orig = winner_shown
    if swapped:
        if winner_shown == "A":
            winner_orig = "B"
        elif winner_shown == "B":
            winner_orig = "A"

    return {
        "winner": winner_orig,          # winner in ORIGINAL order
        "winner_shown": winner_shown,   # winner in SHOWN order (debug)
        "swapped": swapped,
        "reason": reason,
    }

# =========================
# Main: build pairs, run eval, aggregate (C)
# =========================
def aggregate_from_eval_jsonl(eval_jsonl_path: str) -> Dict[Tuple[str, str], dict]:
    """
    Read eval_results.jsonl and compute track x prompt win rates.

    Produces three sets of win rates:
      - *_soft : tie를 0.5/0.5로 분배 (전체 N 기준, 합=1.0)
      - *_strict : tie 제외, 순수 승패만 (decisive N 기준, 합=1.0)
      - tie_rate : tie 비율 (전체 N 기준)
    """
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

def main():
    pairs = build_pairs()
    print(f"[INFO] Total matched pairs: {len(pairs)}")

    # Save pair list
    with open(PAIR_JSONL, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p.__dict__, ensure_ascii=False) + "\n")
    print(f"[INFO] Wrote pairs -> {PAIR_JSONL}")

    pairs_to_eval = pairs[:LIMIT] if LIMIT > 0 else pairs

    # Fresh run: optional (원하면 파일 삭제 후 시작)
    # if os.path.exists(EVAL_JSONL):
    #     os.remove(EVAL_JSONL)

    for p in tqdm(pairs_to_eval, desc="GPT A/B eval"):
        # main에서는 항상 ORIGINAL order 고정: A=base, B=covt
        capA = p.base_caption
        capB = p.covt_caption
        capA_tag = "base"
        capB_tag = "covt"

        # 동일 캡션 사전 필터링: base == covt이면 자동 tie 처리 (API 호출 절감)
        if capA.strip() == capB.strip():
            rec = {
                "track": p.track,
                "prompt_id": p.prompt_id,
                "image_id": p.image_id,
                "image_path": p.image_path,
                "base_caption": p.base_caption,
                "covt_caption": p.covt_caption,
                "capA_tag": capA_tag,
                "capB_tag": capB_tag,
                "capA": capA,
                "capB": capB,
                "swapped": False,
                "shownA_tag": capA_tag,
                "shownB_tag": capB_tag,
                "shownA": capA,
                "shownB": capB,
                "judge_winner": "tie",
                "judge_winner_shown": "tie",
                "winner_model": "tie",
                "reason": "Identical captions: base and covt produced the same text, automatic tie.",
                "judge_model": "skip (identical)",
                "status": "ok",
            }
            with open(EVAL_JSONL, "a", encoding="utf-8") as wf:
                wf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            continue

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                out = judge_ab(p.image_path, capA, capB, prompt_id=p.prompt_id)

                winner = out.get("winner", "tie")               # ORIGINAL A/B space
                winner_shown = out.get("winner_shown", "tie")   # SHOWN A/B space
                swapped = bool(out.get("swapped", False))
                reason = out.get("reason", "")

                # Decide winner_model in terms of base/covt
                if winner == "tie":
                    winner_model = "tie"
                elif winner == "A":
                    winner_model = capA_tag  # base
                elif winner == "B":
                    winner_model = capB_tag  # covt
                else:
                    winner_model = "tie"

                # Determine what the judge SAW (for audit/debug)
                shownA_tag, shownB_tag = capA_tag, capB_tag
                shownA, shownB = capA, capB
                if swapped:
                    shownA_tag, shownB_tag = capB_tag, capA_tag
                    shownA, shownB = capB, capA

                rec = {
                    # identity
                    "track": p.track,
                    "prompt_id": p.prompt_id,
                    "image_id": p.image_id,
                    "image_path": p.image_path,

                    # candidates (normalized, first sentence)
                    "base_caption": p.base_caption,
                    "covt_caption": p.covt_caption,

                    # original A/B fixed mapping used by main
                    "capA_tag": capA_tag,   # always "base"
                    "capB_tag": capB_tag,   # always "covt"
                    "capA": capA,
                    "capB": capB,

                    # shown to judge (after swapped)
                    "swapped": swapped,
                    "shownA_tag": shownA_tag,
                    "shownB_tag": shownB_tag,
                    "shownA": shownA,
                    "shownB": shownB,

                    # judge outputs
                    "judge_winner": winner,               # in ORIGINAL A/B space
                    "judge_winner_shown": winner_shown,   # in SHOWN A/B space
                    "winner_model": winner_model,         # base/covt/tie
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

    # Aggregate win rates track×prompt (read from eval JSONL for robustness)
    agg = aggregate_from_eval_jsonl(EVAL_JSONL)

    with open(SCORES_CSV, "w", newline="", encoding="utf-8") as cf:
        w = csv.writer(cf)
        w.writerow([
            "track", "prompt",
            "num_samples",
            # soft: tie를 0.5/0.5 분배 (합=1.0, 기존 방식)
            "base_winrate_soft", "covt_winrate_soft",
            # strict: tie 제외, 순수 승패만 (합=1.0)
            "base_winrate_strict", "covt_winrate_strict",
            "ties", "tie_rate",
        ])
        for (track, prompt_id), v in sorted(agg.items()):
            n = v["n"]
            bw = v["base_win"]
            cw = v["covt_win"]
            t = v["tie"]
            decisive = bw + cw  # tie 제외 총 승패 수

            w.writerow([
                track, prompt_id, n,
                # soft winrate (tie → 0.5/0.5)
                f"{(bw + t * 0.5) / n:.6f}" if n else "nan",
                f"{(cw + t * 0.5) / n:.6f}" if n else "nan",
                # strict winrate (tie 제외)
                f"{bw / decisive:.6f}" if decisive else "nan",
                f"{cw / decisive:.6f}" if decisive else "nan",
                t,
                f"{t / n:.6f}" if n else "nan",
            ])

    print(f"[DONE] Wrote win rates -> {SCORES_CSV}")

if __name__ == "__main__":
    main()

