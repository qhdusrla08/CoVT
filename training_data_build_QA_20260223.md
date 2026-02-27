# 학습 데이터 구성 QA 및 스크립트 정리

> **작성 일시**: 2026-02-23
> **관련 파일**: `gradio/build_training_data.py`
> **기반 문서**: `arch_kb_experiment_results_20260220_1718.md`

---

## 1. 전략 C (Strategy C) — Step별 상세 설명

`arch_kb_experiment_results_20260220_1718.md` §6-4에서 정의한 전략 C는
**전략 A(GPT winner 선택) + 후처리 중복 제거**의 2단계 파이프라인이다.

### Step 1. GPT Winner 캡션 선택 (전략 A)

`eval_results_P15_kb2_640_4.1.jsonl`의 `winner_model` 필드를 기준으로 per-image 캡션 선택.

```
winner_model == "covt"  → CoVT 캡션 채택
winner_model == "tie"   → CoVT 캡션 채택 (default)
winner_model == "base"  → Base 캡션 채택
```

- 결과: **CoVT 276장 + Tie 81장 + Base 283장 = 640장** (`final_captions_dedup_raw.jsonl`)
- 장점: GPT judge가 이미 검증한 품질 기준으로 최선의 캡션 자동 선택
- 단점: 640쌍 중 283쌍(44.2%)이 Base 채택 → CoVT 학습 비율 감소

### Step 2. 카테고리 내 중복 제거 (sim > 0.85)

같은 카테고리 내에서 `difflib.SequenceMatcher`로 캡션 유사도 계산.

```
sim(cap_i, cap_j) > 0.85  →  단어 수 적은 쪽(덜 구체적인 쪽) 제거
                               동률이면 j 제거
```

- 결과: **640장 → 565장 (−75장)**
- 이 단계가 필요한 이유: Step 1 이후에도 같은 카테고리 내 winner 캡션들끼리 여전히 유사한 경우 존재 (특히 행궁, 팔달문, EXPO 한빛탑)

### Step 3. 빈자리 보충 (optional, `--fill_gaps`)

Step 2에서 제거된 자리에 대해 **탈락한 반대 소스 캡션**이 현재 kept와 충분히 다르면(`sim ≤ 0.85`) 보충 투입.

- 보충 조건: 카테고리 내 모든 kept 캡션과 sim ≤ threshold
- `--fill_gaps` 플래그로 활성화 (기본값: 비활성)

---

## 2. `build_training_data.py` 스크립트

### 2-1. 입력 / 출력

| 구분 | 파일 |
|------|------|
| 입력 (eval) | `eval_outputs_korean/eval_results_P15_kb2_640_4.1.jsonl` |
| 입력 (covt) | `outputs_korean/captions_covt_20_P15.jsonl` |
| 입력 (base) | `outputs_korean/captions_base_20_P15.jsonl` |
| 중간 출력 | `outputs_korean/final_captions_dedup_raw.jsonl` (Step 1 결과, 항상 저장) |
| 최종 출력 | `outputs_korean/final_captions_dedup.jsonl` (Step 2 결과) |

### 2-2. CLI 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--eval` | (필수) | GPT eval results JSONL |
| `--covt` | (필수) | CoVT captions JSONL |
| `--base` | (필수) | Base captions JSONL |
| `--output` | (필수) | 최종 출력 경로 |
| `--sim_threshold` | `0.85` | 중복 판정 유사도 기준 |
| `--no_dedup` | `False` | Step 2 생략 — winner 선택만 실행 |
| `--fill_gaps` | `False` | Step 3 활성화 — 보충 캡션 투입 |

### 2-3. 사용 예

```bash
# 기본 실행 (Step 1 + Step 2)
python build_training_data.py \
  --eval   eval_outputs_korean/eval_results_P15_kb2_640_4.1.jsonl \
  --covt   outputs_korean/captions_covt_20_P15.jsonl \
  --base   outputs_korean/captions_base_20_P15.jsonl \
  --output outputs_korean/final_captions_dedup.jsonl \
  --sim_threshold 0.85

# Step 1만 실행 (winner 선택, dedup 생략)
python build_training_data.py ... --no_dedup

# Step 1 + Step 2 + Step 3 (gap filling 포함)
python build_training_data.py ... --fill_gaps
```

### 2-4. 스크립트 수정 이력

| 시점 | 변경 내용 |
|------|---------|
| 초기 작성 | Step 1 + Step 2 + Step 3 통합 파이프라인 |
| 수정 1 | `--no_dedup` 플래그 추가 — Step 1만 단독 실행 가능 |
| 수정 2 | Step 1 결과를 `_raw.jsonl`로 항상 중간 저장 (검토 및 재활용용) |

---

## 3. 실행 결과 (2026-02-23)

```
[Step 1] GPT winner caption selection
  covt       : 276장
  tie→covt   :  81장
  base       : 283장
  total      : 640장  (CoVT계열 357장 / Base 283장)

[Step 2] Intra-category deduplication (sim > 0.85)
  제거 전 : 640장
  제거 후 : 565장  (−75장)
```

### 주요 중복 제거 카테고리

| 카테고리 | 제거 전 | 제거 후 | 비고 |
|---------|--------|--------|------|
| EXPO Hanbit Tower | 20 | 2 | ⚠ 이미지 자체가 단일 구도 |
| Cheomseongdae observatory | 20 | 8 | ⚠ 수렴 심각 |
| Paldalmun Gate | 20 | 11 | ⚠ 구조적 한계 |
| Haenggung | 20 | 13 | ⚠ KB 개선에도 한계 |
| Gyeongbokgung palace | 20 | 15 | KB v3 효과 일부 |
| Gimbap | 20 | 15 | — |
| Namhansanseong fortress | 20 | 15 | — |

### 최종 소스 비율

| 소스 | 장수 | 비율 |
|------|------|------|
| CoVT계열 | 311장 | 55.0% |
| Base | 254장 | 45.0% |
| **합계** | **565장** | — |

---

## 4. 파인튜닝 데이터 다양성 QA

### Q1. 이미지가 비슷해도 (이미지, 캡션) 쌍이 많으면 좋은 거 아닌가?

**A.** Classification의 데이터 증강과 달리 VLM/멀티모달 파인튜닝에서는 다음과 같이 다르다.

| 상황 | 효과 |
|------|------|
| 다양한 이미지 + 각각 다른 캡션 | ✅ 최고 — 시각·텍스트 다양성 모두 확보 |
| 비슷한 이미지 + 비슷한 캡션 | ⚠ 반복 학습에 가까움 — 제한적 |
| 비슷한 이미지 + 동일 캡션 | ❌ 템플릿 고착 위험 |
| 거의 동일 이미지 + 다른 캡션 | ❌ Inconsistent supervision signal |

- **Classification 증강**: 이미지 변형 + 동일 레이블 → **불변성(invariance) 학습**
- **VLM 파인튜닝**: 비슷한 이미지 + 동일 캡션 → "이 시각 패턴 → 이 텍스트 템플릿" 반복 강화 → 과적합

### Q2. 비슷한 이미지 + 다른 캡션은 괜찮은가?

**A.** 경우에 따라 다르다.

- **이미지의 실제 차이를 반영한 다른 캡션** → ✅ 시각-텍스트 정렬 정확히 학습
- **거의 동일 이미지 + 억지 paraphrase 캡션** → ❌ 같은 시각 입력 → 다른 정답 → Inconsistent gradient → 학습 불안정, hallucination 위험

### Q3. EXPO Hanbit Tower처럼 2장만 남은 카테고리는 어떻게 하나?

**A.** 이미지 데이터셋 자체의 구조적 한계(단일 각도 촬영). 캡션 개선으로 해결 불가.
선택지:
1. 2장 그대로 유지 (학습 기여도 낮음 감수)
2. 전체 데이터셋에서 랜덤 샘플링으로 교체
3. 해당 카테고리를 학습 데이터에서 제외

### Q4. 멀티모달 생성 모델(text↔image 양방향)에서도 같은 원칙이 적용되는가?

**A.** 양방향 모두에서 다양성이 핵심이다.

| 방향 | 비슷한 쌍 반복 시 문제 |
|------|----------------------|
| Image → Text (캡셔닝) | 텍스트 템플릿 고착 |
| Text → Image (이미지 생성) | 특정 구도/스타일만 생성, 생성 다양성 없음 |

- 한국 문화 개념 학습을 위해선 **시각적으로 다양한 예시**가 필수
- 비슷한 이미지 20장 < 다양한 이미지 10장
- 근본 해결책: "첫 20장" 대신 **랜덤 샘플링 + 이미지 다양성 기준 필터링**

---

## 5. 미결 과제

| 과제 | 설명 | 우선순위 |
|------|------|---------|
| 랜덤 샘플링 적용 | 카테고리별 첫 20장 대신 전체에서 랜덤 샘플링으로 시각 다양성 확보 | 높음 |
| ⚠ 카테고리 대책 | EXPO Hanbit Tower(2장), Cheomseongdae(8장) 등 처리 방침 결정 | 중간 |
| History 도메인 | CoVT 전반 약세(35.9%) — KB 보강 vs Base 캡션 활용 | 중간 |
| `--fill_gaps` 효과 검증 | gap filling 적용 전후 카테고리별 장수 및 품질 확인 | 낮음 |
