#!/usr/bin/env python3
"""
build_training_data.py
학습 데이터 구성 스크립트 — 전략 C (전략 A + 후처리 중복 제거)

전략 A : per-image GPT winner 캡션 선택
전략 C : 전략 A + 카테고리 내 SequenceMatcher 중복 제거 (sim > threshold)

사용 예:
  python build_training_data.py \\
    --eval   eval_outputs_korean/eval_results_P15_kb2_640_4.1.jsonl \\
    --covt   outputs_korean/captions_covt_20_P15.jsonl \\
    --base   outputs_korean/captions_base_20_P15.jsonl \\
    --output outputs_korean/final_captions_dedup.jsonl \\
    [--sim_threshold 0.85] \\
    [--fill_gaps]
"""

import argparse
import json
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> list:
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def save_jsonl(items: list, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Step 1: Strategy A — per-image winner selection
# ---------------------------------------------------------------------------

def select_winners(eval_items: list, covt_map: dict) -> tuple:
    """
    eval_items의 winner_model 필드를 기준으로 캡션 선택.
    반환: (raw 데이터 리스트, fallback_map {image_id: 반대 소스 캡션})
    """
    raw = []
    fallback_map = {}  # Step 3 gap filling용 — 탈락한 쪽 캡션
    counts = {"covt": 0, "tie": 0, "base": 0}

    for ev in eval_items:
        iid    = ev["image_id"]
        winner = ev["winner_model"]  # "covt" | "base" | "tie"

        if winner in ("covt", "tie"):
            caption  = ev["covt_caption"]
            source   = winner  # "covt" or "tie"
            fallback = ev["base_caption"]
        else:
            caption  = ev["base_caption"]
            source   = "base"
            fallback = ev["covt_caption"]

        counts[winner if winner in counts else "base"] += 1

        # category는 covt caption 파일에서 가져옴
        category = covt_map[iid]["category"] if iid in covt_map else "unknown"

        raw.append({
            "image_id":   iid,
            "image_path": ev["image_path"],
            "category":   category,
            "caption":    caption,
            "source":     source,
        })
        fallback_map[iid] = fallback

    return raw, fallback_map, counts


# ---------------------------------------------------------------------------
# Step 2: Intra-category deduplication
# ---------------------------------------------------------------------------

def _sim(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def dedup_category(items: list, threshold: float, fallback_map: dict) -> tuple:
    """
    카테고리 내 중복 제거.
    - sim > threshold인 쌍에서 단어 수 적은 캡션을 제거
    - 제거된 자리에 대해 fallback 캡션이 kept와 충분히 다르면 gap_candidate로 반환

    반환: (kept 리스트, gap_candidates 리스트)
    """
    n = len(items)
    removed = set()

    for i in range(n):
        for j in range(i + 1, n):
            if i in removed or j in removed:
                continue
            s = _sim(items[i]["caption"], items[j]["caption"])
            if s > threshold:
                wi = len(items[i]["caption"].split())
                wj = len(items[j]["caption"].split())
                # 단어 수 적은(덜 구체적인) 쪽 제거; 동률이면 j 제거
                drop = i if wi < wj else j
                removed.add(drop)

    kept = [item for idx, item in enumerate(items) if idx not in removed]
    dropped = [items[idx] for idx in sorted(removed)]

    # gap_candidates: fallback 캡션이 kept와 충분히 다른 경우만 후보로
    gap_candidates = []
    for item in dropped:
        iid = item["image_id"]
        if iid not in fallback_map:
            continue
        fb_caption = fallback_map[iid]
        if all(_sim(fb_caption, k["caption"]) <= threshold for k in kept):
            fb_source = "base" if item["source"] in ("covt", "tie") else "covt"
            gap_candidates.append({
                "image_id":   iid,
                "image_path": item["image_path"],
                "category":   item["category"],
                "caption":    fb_caption,
                "source":     fb_source,
                "gap_fill":   True,
            })

    return kept, gap_candidates


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build final training data: Strategy A + dedup (Strategy C)"
    )
    parser.add_argument("--eval",          required=True,
                        help="GPT eval results JSONL (winner_model 필드 포함)")
    parser.add_argument("--covt",          required=True,
                        help="CoVT captions JSONL (category 필드 참조용)")
    parser.add_argument("--base",          required=True,
                        help="Base captions JSONL")
    parser.add_argument("--output",        required=True,
                        help="출력 JSONL 경로")
    parser.add_argument("--sim_threshold", type=float, default=0.85,
                        help="중복 판정 유사도 기준 (default: 0.85)")
    parser.add_argument("--fill_gaps",     action="store_true",
                        help="Step 3: 제거된 자리를 반대 소스 캡션으로 보충 (optional)")
    parser.add_argument("--no_dedup",      action="store_true",
                        help="Step 2 생략: winner 선택(Step 1)만 실행하고 raw 결과 저장")
    args = parser.parse_args()

    # ── Load ──────────────────────────────────────────────────────────────
    print(f"[Load] eval : {args.eval}")
    eval_items = load_jsonl(args.eval)
    print(f"[Load] covt : {args.covt}")
    covt_items = load_jsonl(args.covt)
    print(f"[Load] base : {args.base}")
    base_items = load_jsonl(args.base)  # 현재는 category 참조 목적으로만 로드

    covt_map = {item["image_id"]: item for item in covt_items}

    # ── Step 1: Strategy A ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"[Step 1] GPT winner caption selection (Strategy A)")
    print(f"{'='*60}")

    raw, fallback_map, step1_counts = select_winners(eval_items, covt_map)

    covt_total = step1_counts["covt"] + step1_counts["tie"]
    print(f"  covt       : {step1_counts['covt']:>3}장")
    print(f"  tie→covt   : {step1_counts['tie']:>3}장")
    print(f"  base       : {step1_counts['base']:>3}장")
    print(f"  ─────────────────────")
    print(f"  total      : {len(raw):>3}장  (CoVT계열 {covt_total}장 / Base {step1_counts['base']}장)")

    # Step 1 결과 항상 _raw.jsonl로 저장 (중간 결과 보존)
    raw_output = args.output.replace(".jsonl", "_raw.jsonl")
    save_jsonl(raw, raw_output)
    print(f"\n  [중간 저장] {raw_output}  ({len(raw)}장)")

    if args.no_dedup:
        print(f"\n[--no_dedup 설정] Step 2 생략 — raw 결과를 최종 출력으로 저장")
        save_jsonl(raw, args.output)
        print(f"[Output] {args.output}  ({len(raw)}장 저장 완료)")
        return

    # ── Step 2: Intra-category deduplication ─────────────────────────────
    print(f"\n{'='*60}")
    print(f"[Step 2] Intra-category deduplication  (sim > {args.sim_threshold})")
    print(f"{'='*60}")

    by_category = defaultdict(list)
    for item in raw:
        by_category[item["category"]].append(item)

    deduped = []
    all_gap_candidates = []
    dedup_log = []

    for cat in sorted(by_category.keys()):
        items = by_category[cat]
        kept, gap_cands = dedup_category(items, args.sim_threshold, fallback_map)
        removed_n = len(items) - len(kept)
        deduped.extend(kept)
        all_gap_candidates.extend(gap_cands)
        if removed_n > 0:
            dedup_log.append((cat, len(items), len(kept), removed_n))

    total_removed = len(raw) - len(deduped)
    print(f"  제거 전 : {len(raw)}장")
    print(f"  제거 후 : {len(deduped)}장  (−{total_removed}장)")

    if dedup_log:
        print(f"\n  카테고리별 중복 제거 현황:")
        print(f"  {'카테고리':<35} {'before':>6}  {'after':>5}  {'removed':>7}")
        print(f"  {'-'*56}")
        for cat, before, after, removed in dedup_log:
            print(f"  {cat:<35} {before:>6}  {after:>5}  −{removed:>6}")
    else:
        print("  중복 없음 — 제거된 항목 없음")

    # ── Step 3: Gap filling (optional) ───────────────────────────────────
    final = deduped
    if args.fill_gaps:
        print(f"\n{'='*60}")
        print(f"[Step 3] Gap filling (optional)")
        print(f"{'='*60}")
        if all_gap_candidates:
            print(f"  보충 후보  : {len(all_gap_candidates)}장")
            final = deduped + all_gap_candidates
            print(f"  보충 후 총 : {len(final)}장")
        else:
            print(f"  보충 가능한 후보 없음")

    # ── Save ──────────────────────────────────────────────────────────────
    save_jsonl(final, args.output)

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"[Summary]")
    print(f"{'='*60}")

    src_counts = defaultdict(int)
    cat_counts  = defaultdict(int)
    for item in final:
        src = item["source"]
        src_counts["covt" if src in ("covt", "tie") else src] += 1
        cat_counts[item["category"]] += 1

    print(f"  소스별:")
    print(f"    CoVT계열 : {src_counts['covt']:>3}장  ({src_counts['covt']/len(final)*100:.1f}%)")
    print(f"    Base     : {src_counts['base']:>3}장  ({src_counts['base']/len(final)*100:.1f}%)")
    print(f"    합계     : {len(final):>3}장")

    print(f"\n  카테고리별 (20장 → 제거 후):")
    for cat in sorted(cat_counts.keys()):
        cnt = cat_counts[cat]
        marker = " ⚠" if cnt < 15 else ""
        print(f"    {cat:<35} {cnt:>2}장{marker}")

    print(f"\n[Output] {args.output}  ({len(final)}장 저장 완료)")


if __name__ == "__main__":
    main()
