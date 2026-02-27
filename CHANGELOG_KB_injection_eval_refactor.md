# Changelog: KB 주입 이후 평가 체계 리팩토링

> 작성일: 2026-02-11
> 대상 파일: `caption_gen_korean.py`, `gpt_eval_korean.py`, `gpt_eval.py`
> 배경: P7 tie case 분석을 통해 평가 기준과 캡션 생성 프롬프트의 구조적 문제를 식별하고 개선

---

## 1. P7 Tie Case 분석 결과 (동기)

기존 P7 평가 결과: **CoVT 56% / Base 34% / Tie 10%** (50건 중 tie 5건)

### Tie 발생 패턴 3가지

| 패턴 | 해당 케이스 | 원인 |
|---|---|---|
| 시각적 검증 불가한 역사적 주장 | 독립운동, 남한산성, 한국전쟁 | 양쪽 모두 역사적 사실을 언급했으나 Judge가 "검증 불가"로 동점 처리 |
| 동일 수준의 모호한 문화 설명 | 성묘 | 이미지 자체가 문화 맥락을 드러내지 못해 양쪽 모두 낮은 점수 |
| 캡션 동일 (지식 주입 실패) | 윷놀이 | base와 covt가 완전히 동일한 텍스트 생성 |

### 핵심 모순 발견

KB(`KNOWLEDGE_BASE`)에서 정확한 문화 사실(날짜, 왕조, UNESCO 등)을 P7 프롬프트로 주입하는데, 기존 Judge가 이를 **"시각적으로 검증 불가한 주장"으로 패널티** 부과 → CoVT의 지식 주입이 잘 될수록 오히려 불리해지는 자기모순 구조.

---

## 2. 수정 사항 요약

### 2-1. 평가 기준 분리: 프롬프트 그룹별 Judge System (`gpt_eval_korean.py`)

**변경 전**: 단일 `JUDGE_SYSTEM`으로 P0~P7 전체 평가
**변경 후**: 3개 그룹별 특화 Judge System + `prompt_id` 기반 자동 선택

```
prompt_id  →  _PROMPT_GROUP  →  Judge System + User Text
──────────────────────────────────────────────────────────
P0~P3      →  "visual"       →  JUDGE_SYSTEM_VISUAL
P4~P5      →  "technical"    →  JUDGE_SYSTEM_TECHNICAL
P6~P7      →  "cultural"     →  JUDGE_SYSTEM_CULTURAL
```

#### Group A: `JUDGE_SYSTEM_VISUAL` (P0~P3)

시각 묘사 품질 중심. 카테고리 인식은 보상하되 문화적 깊이는 요구하지 않음.

| 우선순위 | 기준 | 설명 |
|---|---|---|
| 1 | Visual Grounding & Accuracy | 보이는 것을 정확하게 묘사했는가 |
| 2 | Specificity & Disambiguation | 구체적 시각 디테일 (색상, 형태, 재질) |
| 3 | Neutrality & Objectivity | 주관적/감정적 표현 패널티 |
| 4 | Absence of Hallucination | 시각적 날조 패널티 (정확한 한국 용어는 허용) |
| 5 | Training Usefulness | 간결하고 정보 밀도 높은 캡션 |

#### Group B: `JUDGE_SYSTEM_TECHNICAL` (P4~P5)

시각 기술 지표 활용 평가. segmentation/depth/patch 특성의 반영도 추가 평가.

| 우선순위 | 기준 | 설명 |
|---|---|---|
| 1 | Visual Grounding with Structural Precision | 시각 구조(전경/배경, 깊이, 영역 분리) 반영 |
| 2 | Specificity & Disambiguation | 정밀한 시각 속성 (크기, 질감, 패턴) |
| 3 | Neutrality & Objectivity | 주관적 표현 패널티 |
| 4 | Absence of Hallucination | 시각적 날조 패널티 |
| 5 | Training Usefulness | 간결성 및 정보 밀도 |

#### Group C: `JUDGE_SYSTEM_CULTURAL` (P6~P7)

시각-문화 연결(Visual-Cultural Bridging) 중심. 정확한 문화 지식은 보상, 시각적 날조만 패널티.

| 우선순위 | 기준 | 설명 |
|---|---|---|
| 1 | Visual-Cultural Bridging | 시각 요소 → 문화적 의미로의 연결 품질 |
| 2 | Korean Knowledge Injection Value | 한국 고유 용어, 정확한 역사/문화 사실 (가점) |
| 3 | Specificity & Disambiguation | 유사 개념 구분 (Jesa vs Seongmyo 등) |
| 4 | Cultural Accuracy & Terminology | 틀린 사실만 패널티, 정확한 사실 ≠ 패널티 |
| 5 | Training Usefulness | 시각-문화 연관 학습에 유용한 캡션 |

**Hallucination vs. Knowledge 구분 명시:**
- **Hallucination** (패널티): 이미지에 없는 시각적 요소 묘사
- **Knowledge** (보상): 보이는 장면에 정확한 문화적 맥락 연결

### 2-2. 공통 Decision Rules (`_JUDGE_COMMON_DECISION_RULES`)

3개 그룹 모두에 적용되는 판정 규칙:

- **Cascading Tiebreaker**: Criterion 1→2→3→4→5 순차 적용, 차이 발생 시 즉시 판정
- **Micro Tiebreaker**: (a) 시각 디테일 수, (b) 간결성
- **Tie 제한**: 캡션이 거의 동일한 경우에만 tie 허용, 단순 유사성은 tie 불가

### 2-3. 동일 캡션 사전 필터링 (`gpt_eval_korean.py`)

```python
if capA.strip() == capB.strip():
    # 자동 tie 처리, GPT API 호출 생략
    judge_model = "skip (identical)"
```

- base와 covt 캡션이 동일한 경우 API 호출 없이 자동 tie
- `judge_model: "skip (identical)"`로 구분 가능
- 불필요한 API 비용 절감

### 2-4. `judge_ab()` 함수 시그니처 변경 (`gpt_eval_korean.py`)

```python
# 변경 전
def judge_ab(image_path: str, capA: str, capB: str) -> dict:

# 변경 후
def judge_ab(image_path: str, capA: str, capB: str, prompt_id: str = "P7") -> dict:
```

- `prompt_id`를 받아 `get_judge_config()`으로 그룹별 Judge System 및 User Text 자동 선택
- `main()` 루프에서 `prompt_id=p.prompt_id` 전달

---

## 3. P7 프롬프트 수정 (`caption_gen_korean.py`)

### 변경 내용

```python
# 변경 전 (병렬 나열)
"P7": "Describe the visual appearance and cultural context of '{category}' ({definition}) in **exactly one sentence**."

# 변경 후 (인과적 연결)
"P7": "Describe what is visible in this image and explain how it relates to '{category}' ({definition}) in **exactly one sentence**."
```

### 변경 근거

| | 기존 | 수정 |
|---|---|---|
| 구조 | "visual appearance **and** cultural context" | "what is visible ... **and explain how it relates to**" |
| 모델 행동 | 시각 묘사 + 문화 설명을 **따로 나열** | 시각 요소 → 문화 맥락을 **인과적으로 연결** |
| Judge 정합성 | JUDGE_SYSTEM_CULTURAL의 Criterion 1 (Visual-Cultural Bridging)에 직접 대응 |

### 렌더링 예시

```
Describe what is visible in this image and explain how it relates to
'Ganggangsullae' (A Korean circle dance and folk song performed by women
under the full moon, designated as UNESCO Intangible Cultural Heritage.)
in exactly one sentence.
```

---

## 4. Win Rate CSV 계산 방식 수정 (`gpt_eval_korean.py`, `gpt_eval.py`)

### 변경 내용

`aggregate_from_eval_jsonl()` 및 CSV 출력부를 양쪽 파일 모두 동일하게 수정.

**변경 전**: tie를 0.5/0.5로 분배하여 단일 승률만 출력

```csv
track,prompt,num_samples,base_winrate,covt_winrate,ties
B,P7,50,0.390000,0.610000,5
```

**변경 후**: soft/strict 승률 + tie 비율을 분리 출력

```csv
track,prompt,num_samples,base_winrate_soft,covt_winrate_soft,base_winrate_strict,covt_winrate_strict,ties,tie_rate
B,P7,50,0.390000,0.610000,0.377778,0.622222,5,0.100000
```

### 컬럼 정의

| 컬럼 | 계산 방식 | 설명 |
|---|---|---|
| `base_winrate_soft` | `(base_win + tie×0.5) / N` | tie를 0.5/0.5 분배 (기존 방식, 하위호환) |
| `covt_winrate_soft` | `(covt_win + tie×0.5) / N` | tie를 0.5/0.5 분배 |
| `base_winrate_strict` | `base_win / (base_win + covt_win)` | tie 제외, 순수 승패만 (실제 우열) |
| `covt_winrate_strict` | `covt_win / (base_win + covt_win)` | tie 제외, 순수 승패만 |
| `ties` | 건수 | tie 발생 횟수 |
| `tie_rate` | `tie / N` | tie 비율 (평가 품질 지표) |

### P7 데이터 검증

| 지표 | 기존 (soft only) | 수정 후 (strict) | 차이 |
|---|---|---|---|
| base | 0.3900 | **0.3778** | tie 제거로 실제 열세 더 드러남 |
| covt | 0.6100 | **0.6222** | tie 제거로 실제 우세 더 드러남 |
| tie_rate | (건수만) | **0.1000** | 10% tie 발생 비율 가시화 |

---

## 5. 수정 파일 목록

| 파일 | 수정 내용 |
|---|---|
| `caption_gen_korean.py` | P7 프롬프트를 시각-문화 bridging 지시로 변경 |
| `gpt_eval_korean.py` | Judge System 3그룹 분리, 동일 캡션 필터링, judge_ab 시그니처 변경, win rate 계산 수정 |
| `gpt_eval.py` | win rate 계산 방식 동일하게 수정 (soft/strict/tie_rate) |
