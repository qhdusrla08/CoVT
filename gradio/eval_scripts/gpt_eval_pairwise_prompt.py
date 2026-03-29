"""
Pairwise Prompt Comparison Eval Script
---------------------------------------
서로 다른 프롬프트의 CoVT 캡션끼리 직접 비교하여
어떤 프롬프트가 가장 높은 품질의 KB 주입 캡션을 생성하는지 판별.

대상: P13, P16, P17, P19 → C(4,2)=6쌍 × 50이미지 = 300 API calls
"""

import os
import re
import json
import csv
import time
import base64
import random
import itertools
from typing import Dict, List, Tuple, Set

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
LIMIT = int(os.getenv("EVAL_LIMIT", "0"))  # per matchup; 0 = all

RANDOM_SEED = int(os.getenv("EVAL_SEED", "42"))
random.seed(RANDOM_SEED)

MAX_RETRIES = 3
SLEEP_BETWEEN_CALLS_SEC = float(os.getenv("SLEEP_SEC", "0.0"))

TARGET_PROMPTS = ["P15", "P19"]

EVAL_JSONL = os.path.join(EVAL_DIR, "eval_results_pairwise_prompt_1519.jsonl")
SCORES_CSV = os.path.join(EVAL_DIR, "win_rates_pairwise_prompt_1519.csv")

client = OpenAI()

# =========================
# Reused utilities from gpt_eval_korean.py
# =========================
def normalize_caption(c: str) -> str:
    if not c:
        return ""
    c = c.replace("<answer>", "").replace("</answer>", "").strip()
    c = c.replace("\n", " ").strip()
    c = re.sub(r"<think>.*?</think>", "", c, flags=re.DOTALL).strip()
    return c


def image_to_data_url(image_path: str, max_side: int = 1024) -> str:
    import io
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        nw, nh = int(w * scale), int(h * scale)
        img = img.resize((nw, nh), Image.BICUBIC)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def safe_json_load(s: str) -> dict:
    s = (s or "").strip()
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if m:
        s = m.group(0)
    return json.loads(s)


# =========================
# Judge system prompt (same as gpt_eval_korean.py)
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


# =========================
# Load caption files
# =========================
def load_covt_captions(prompt_id: str) -> Dict[str, dict]:
    """Load covt caption file, return dict keyed by image_id."""
    path = os.path.join(OUTPUT_DIR, f"captions_covt_20_{prompt_id}.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Caption file not found: {path}")

    captions = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("status") != "ok":
                continue
            image_id = r.get("image_id")
            if not image_id:
                continue
            cap_norm = normalize_caption(r.get("caption", ""))
            if not cap_norm:
                continue
            captions[image_id] = {
                "image_id": image_id,
                "image_path": r.get("image_path", ""),
                "track": r.get("track", "B"),
                "caption": cap_norm,
            }
    return captions


# =========================
# Resume: load already-evaluated (matchup, image_id) pairs
# =========================
def load_done_keys(eval_jsonl_path: str) -> Set[Tuple[str, str]]:
    done = set()
    if not os.path.exists(eval_jsonl_path):
        return done
    with open(eval_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            matchup = r.get("matchup", "")
            image_id = r.get("image_id", "")
            if matchup and image_id:
                done.add((matchup, image_id))
    return done


# =========================
# Judge call
# =========================
def judge_ab(image_path: str, capA: str, capB: str) -> dict:
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

    # Map winner back to ORIGINAL (capA=prompt_a, capB=prompt_b) order
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
        "shownA": shownA,
        "shownB": shownB,
        "reason": reason,
    }


# =========================
# Aggregation
# =========================
def aggregate_from_eval_jsonl(eval_jsonl_path: str) -> Dict[str, dict]:
    """Read eval JSONL and compute per-matchup win rates."""
    agg: Dict[str, dict] = {}
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

            matchup = r.get("matchup")
            winner_prompt = r.get("winner_prompt")
            prompt_a = r.get("prompt_a")
            prompt_b = r.get("prompt_b")
            if not matchup or not winner_prompt:
                continue

            if matchup not in agg:
                agg[matchup] = {
                    "prompt_a": prompt_a,
                    "prompt_b": prompt_b,
                    "n": 0,
                    "a_win": 0,
                    "b_win": 0,
                    "tie": 0,
                }

            agg[matchup]["n"] += 1
            if winner_prompt == prompt_a:
                agg[matchup]["a_win"] += 1
            elif winner_prompt == prompt_b:
                agg[matchup]["b_win"] += 1
            else:
                agg[matchup]["tie"] += 1

    return agg


def write_scores_csv(agg: Dict[str, dict], csv_path: str):
    with open(csv_path, "w", newline="", encoding="utf-8") as cf:
        w = csv.writer(cf)
        w.writerow([
            "matchup", "prompt_a", "prompt_b", "num_samples",
            "a_winrate_soft", "b_winrate_soft",
            "a_winrate_strict", "b_winrate_strict",
            "ties", "tie_rate",
        ])
        for matchup, v in sorted(agg.items()):
            n = v["n"]
            aw = v["a_win"]
            bw = v["b_win"]
            t = v["tie"]
            decisive = aw + bw

            w.writerow([
                matchup, v["prompt_a"], v["prompt_b"], n,
                f"{(aw + t * 0.5) / n:.4f}" if n else "nan",
                f"{(bw + t * 0.5) / n:.4f}" if n else "nan",
                f"{aw / decisive:.4f}" if decisive else "nan",
                f"{bw / decisive:.4f}" if decisive else "nan",
                t,
                f"{t / n:.4f}" if n else "nan",
            ])


def compute_elo_and_summary(agg: Dict[str, dict]):
    """Compute overall head-to-head summary table."""
    # Build win matrix
    prompts = sorted(TARGET_PROMPTS)
    wins = {p: {q: 0 for q in prompts} for p in prompts}
    total = {p: {q: 0 for q in prompts} for p in prompts}

    for matchup, v in agg.items():
        pa, pb = v["prompt_a"], v["prompt_b"]
        n = v["n"]
        wins[pa][pb] += v["a_win"]
        wins[pb][pa] += v["b_win"]
        # ties count as 0.5 each
        wins[pa][pb] += v["tie"] * 0.5
        wins[pb][pa] += v["tie"] * 0.5
        total[pa][pb] += n
        total[pb][pa] += n

    print("\n=== Head-to-Head Win Rates (row wins vs col) ===")
    header = f"{'':>6}" + "".join(f"{p:>8}" for p in prompts)
    print(header)
    for p in prompts:
        row = f"{p:>6}"
        for q in prompts:
            if p == q:
                row += f"{'---':>8}"
            elif total[p][q] > 0:
                wr = wins[p][q] / total[p][q]
                row += f"{wr:>8.1%}"
            else:
                row += f"{'N/A':>8}"
        print(row)

    # Overall win rate per prompt
    print("\n=== Overall Win Rate (soft, across all matchups) ===")
    for p in prompts:
        total_wins = sum(wins[p][q] for q in prompts if q != p)
        total_games = sum(total[p][q] for q in prompts if q != p)
        if total_games > 0:
            print(f"  {p}: {total_wins/total_games:.1%} ({total_wins:.0f}/{total_games})")


# =========================
# Main
# =========================
def main():
    # Load all caption data
    print(f"[INFO] Loading CoVT captions for prompts: {TARGET_PROMPTS}")
    all_captions: Dict[str, Dict[str, dict]] = {}
    for pid in TARGET_PROMPTS:
        all_captions[pid] = load_covt_captions(pid)
        print(f"  {pid}: {len(all_captions[pid])} images")

    # Find common image_ids across ALL target prompts
    common_ids = None
    for pid in TARGET_PROMPTS:
        ids = set(all_captions[pid].keys())
        common_ids = ids if common_ids is None else common_ids & ids
    common_ids = sorted(common_ids)
    print(f"[INFO] Common images across all prompts: {len(common_ids)}")

    # Generate all matchup pairs: C(4,2) = 6
    matchups = list(itertools.combinations(TARGET_PROMPTS, 2))
    print(f"[INFO] Matchups: {[f'{a}_vs_{b}' for a, b in matchups]}")

    # Build eval tasks
    tasks = []
    for pa, pb in matchups:
        matchup_name = f"{pa}_vs_{pb}"
        for image_id in common_ids:
            tasks.append({
                "matchup": matchup_name,
                "prompt_a": pa,
                "prompt_b": pb,
                "image_id": image_id,
                "image_path": all_captions[pa][image_id]["image_path"],
                "track": all_captions[pa][image_id]["track"],
                "caption_a": all_captions[pa][image_id]["caption"],
                "caption_b": all_captions[pb][image_id]["caption"],
            })

    print(f"[INFO] Total eval tasks: {len(tasks)}")

    # Resume: skip already evaluated
    done_keys = load_done_keys(EVAL_JSONL)
    tasks_remaining = [t for t in tasks if (t["matchup"], t["image_id"]) not in done_keys]
    if done_keys:
        print(f"[INFO] Resuming: {len(done_keys)} already done, {len(tasks_remaining)} remaining")

    # Apply limit per matchup if set
    if LIMIT > 0:
        limited = []
        matchup_counts: Dict[str, int] = {}
        for t in tasks_remaining:
            mu = t["matchup"]
            matchup_counts.setdefault(mu, 0)
            if matchup_counts[mu] < LIMIT:
                limited.append(t)
                matchup_counts[mu] += 1
        tasks_remaining = limited
        print(f"[INFO] Limited to {LIMIT} per matchup: {len(tasks_remaining)} tasks")

    # Run evaluations
    for t in tqdm(tasks_remaining, desc="Pairwise prompt eval"):
        capA = t["caption_a"]
        capB = t["caption_b"]

        # Identical captions → auto tie
        if capA.strip() == capB.strip():
            rec = {
                "track": t["track"],
                "matchup": t["matchup"],
                "prompt_a": t["prompt_a"],
                "prompt_b": t["prompt_b"],
                "image_id": t["image_id"],
                "image_path": t["image_path"],
                "caption_a": capA,
                "caption_b": capB,
                "swapped": False,
                "shownA": capA,
                "shownB": capB,
                "judge_winner": "tie",
                "winner_prompt": "tie",
                "reason": "Identical captions from both prompts, automatic tie.",
                "judge_model": "skip (identical)",
                "status": "ok",
            }
            with open(EVAL_JSONL, "a", encoding="utf-8") as wf:
                wf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            continue

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                out = judge_ab(t["image_path"], capA, capB)

                winner = out["winner"]  # A or B or tie (original order)
                swapped = out["swapped"]
                reason = out["reason"]

                # Map A/B winner to prompt name
                if winner == "A":
                    winner_prompt = t["prompt_a"]
                elif winner == "B":
                    winner_prompt = t["prompt_b"]
                else:
                    winner_prompt = "tie"

                rec = {
                    "track": t["track"],
                    "matchup": t["matchup"],
                    "prompt_a": t["prompt_a"],
                    "prompt_b": t["prompt_b"],
                    "image_id": t["image_id"],
                    "image_path": t["image_path"],
                    "caption_a": capA,
                    "caption_b": capB,
                    "swapped": swapped,
                    "shownA": out["shownA"],
                    "shownB": out["shownB"],
                    "judge_winner": winner,
                    "winner_prompt": winner_prompt,
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
                        "track": t["track"],
                        "matchup": t["matchup"],
                        "prompt_a": t["prompt_a"],
                        "prompt_b": t["prompt_b"],
                        "image_id": t["image_id"],
                        "image_path": t["image_path"],
                        "status": "fail",
                        "error": repr(e),
                        "judge_model": JUDGE_MODEL,
                    }
                    with open(EVAL_JSONL, "a", encoding="utf-8") as wf:
                        wf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                else:
                    time.sleep(1.0 * attempt)

    print(f"[INFO] Wrote eval logs -> {EVAL_JSONL}")

    # Aggregate and write CSV
    agg = aggregate_from_eval_jsonl(EVAL_JSONL)
    write_scores_csv(agg, SCORES_CSV)
    print(f"[DONE] Wrote win rates -> {SCORES_CSV}")

    # Print summary
    compute_elo_and_summary(agg)


if __name__ == "__main__":
    main()
