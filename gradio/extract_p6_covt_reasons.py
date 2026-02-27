"""
P6 프롬프트에서 CoVT가 승리한 평가 건의 GPT reason을 추출·분석하는 스크립트.

사용법:
    python extract_p6_covt_reasons.py
"""

import json
import os
import re
from collections import Counter

EVAL_JSONL = os.path.join("eval_outputs_korean", "eval_results.jsonl")
OUTPUT_PATH = os.path.join("eval_outputs_korean", "p6_covt_win_reasons.jsonl")
SUMMARY_PATH = os.path.join("eval_outputs_korean", "p6_covt_win_summary.txt")

TARGET_PROMPT = "P6"


def load_p6_covt_wins(eval_path: str) -> list[dict]:
    results = []
    with open(eval_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if (
                rec.get("prompt_id") == TARGET_PROMPT
                and rec.get("winner_model") == "covt"
                and rec.get("status") == "ok"
            ):
                results.append(rec)
    return results


def extract_reasons(records: list[dict]) -> list[dict]:
    extracted = []
    for rec in records:
        extracted.append({
            "image_id": rec.get("image_id", ""),
            "image_path": rec.get("image_path", ""),
            "base_caption": rec.get("base_caption", ""),
            "covt_caption": rec.get("covt_caption", ""),
            "reason": rec.get("reason", ""),
        })
    return extracted


def collect_korean_terms(extracted: list[dict]) -> Counter:
    """reason + covt_caption에서 한국 문화 용어를 동적으로 추출하여 빈도를 센다."""

    # data/B_korean 디렉토리명에서 카테고리 용어 자동 수집
    korean_terms: dict[str, list[str]] = {}  # display_name -> [lowercase variants]
    # 일반 영어 단어와 겹쳐 false positive가 발생하는 서브토큰 제외
    generic_tokens = {
        "the", "of", "and", "in", "at", "on", "for", "to", "a", "an",
        "traditional", "wedding", "independence", "movement", "hall",
        "korea", "korean", "war", "new", "year", "bow", "deep", "gate",
        "tower", "palace", "observatory", "fortress", "kite", "flying",
        "mask", "dance", "admiral", "soup", "fried", "battered", "meatballs",
    }
    data_dir = os.path.join("data", "B_korean")
    if os.path.isdir(data_dir):
        for dname in sorted(os.listdir(data_dir)):
            # "9001_Kimjang" -> "Kimjang"
            parts = dname.split("_", 1)
            if len(parts) < 2:
                continue
            raw_name = parts[1]  # e.g. "Tteokguk_Mandu soup"
            # 전체 이름은 항상 포함
            variants = [raw_name.lower().replace("_", " ")]
            # 서브토큰 중 한국 고유어만 추가 (일반 영어 단어 제외)
            tokens = re.split(r"[_ ]+", raw_name)
            for tok in tokens:
                if len(tok) >= 4 and tok.lower() not in generic_tokens:
                    variants.append(tok.lower())
            korean_terms[raw_name.replace("_", " ")] = list(set(variants))

    # 데이터 디렉토리 외 reason에서 자주 등장하는 추가 용어
    extra_terms = {
        "hanbok": ["hanbok"],
        "hanok": ["hanok"],
        "kimchi": ["kimchi"],
        "banchan": ["banchan"],
        "nori seaweed": ["nori seaweed", "nori"],
        "tteok (rice cake)": ["tteok", "rice cake"],
        "Lunar New Year": ["lunar new year"],
        "taegukgi": ["taegukgi", "taegeukgi"],
        "ondol": ["ondol"],
        "jangdokdae": ["jangdokdae"],
        "Hangul": ["hangul"],
        "UNESCO": ["unesco"],
    }
    korean_terms.update(extra_terms)

    counts = Counter()
    for item in extracted:
        text = (item["reason"] + " " + item["covt_caption"]).lower()
        for display_name, variants in korean_terms.items():
            if any(v in text for v in variants):
                counts[display_name] += 1

    return counts


def build_summary(records: list[dict], extracted: list[dict]) -> str:
    lines = []
    lines.append(f"=== P6 CoVT 승리 reason 분석 ===\n")
    lines.append(f"총 P6 평가 건수 (eval_results.jsonl 내): 아래 통계에서 확인")
    lines.append(f"CoVT 승리 건수: {len(records)}\n")

    n = len(extracted)

    # --- (A) 평가 기준 키워드 빈도 ---
    criteria_groups = {
        "visual grounding": [
            "visually grounded", "visual grounding", "visually supported",
            "visually verif", "directly observable", "clearly visible",
        ],
        "specific/detail": [
            "specific", "detail", "detailed", "specificity",
            "disambiguation", "disambiguat",
        ],
        "Korean knowledge injection": [
            "korean knowledge", "korean cultural", "korean-specific",
            "korean domain", "korean culture", "cultural knowledge",
            "knowledge injection",
        ],
        "hallucination penalty": [
            "hallucin", "unverifiable", "unverif", "unsupported",
            "not visually supported", "not visually verif",
            "not directly observable", "overreach",
            "not clearly supported",
        ],
        "concise/conciseness": ["concise", "conciseness"],
        "accurate/accuracy": ["accurate", "accuracy", "precise", "precision"],
        "terminology quality": ["terminology", "cultural term", "korean term"],
        "training usefulness": ["training", "fine-tun", "suitable for"],
    }

    criteria_counts = Counter()
    for item in extracted:
        reason_lower = item["reason"].lower()
        for group_name, keywords in criteria_groups.items():
            if any(kw in reason_lower for kw in keywords):
                criteria_counts[group_name] += 1

    lines.append("--- (A) 평가 기준 키워드 빈도 (reason 내 등장 건수) ---")
    for kw, cnt in criteria_counts.most_common():
        lines.append(f"  {kw:30s}: {cnt:3d} / {n} ({cnt / n * 100:.1f}%)")

    # --- (B) 한국 문화 용어 빈도 (동적 추출) ---
    korean_counts = collect_korean_terms(extracted)

    lines.append("")
    lines.append("--- (B) 한국 문화 용어 등장 빈도 (reason + covt_caption) ---")
    for term, cnt in korean_counts.most_common():
        lines.append(f"  {term:30s}: {cnt:3d} / {n} ({cnt / n * 100:.1f}%)")

    lines.append("")
    lines.append("--- 개별 reason 목록 ---")
    for i, item in enumerate(extracted, 1):
        lines.append(f"\n[{i}] image_id: {item['image_id']}")
        lines.append(f"    base: {item['base_caption'][:120]}...")
        lines.append(f"    covt: {item['covt_caption'][:120]}...")
        lines.append(f"    reason: {item['reason']}")

    return "\n".join(lines)


def main():
    records = load_p6_covt_wins(EVAL_JSONL)
    print(f"[INFO] P6 CoVT 승리 건수: {len(records)}")

    extracted = extract_reasons(records)

    # JSONL 저장
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for item in extracted:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"[INFO] 개별 reason -> {OUTPUT_PATH}")

    # 요약 텍스트 저장
    summary = build_summary(records, extracted)
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write(summary)
    print(f"[INFO] 요약 분석  -> {SUMMARY_PATH}")
    print()
    print(summary)


if __name__ == "__main__":
    main()
