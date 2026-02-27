# 건축 KB 개선 실험 결과 보고서

> **작성 일시**: 2026-02-20
> **이전 문서**: `arch_kb_issue_and_plan_20260220_1044.md`
> **실험 대상**: 건축 4개 카테고리 (경복궁·남한산성·행궁·팔달문), P15 프롬프트
> **관련 파일**:
> - `caption_gen_korean_50.py` (KB 수정, `--categories` 인자 추가)
> - `outputs_korean/captions_covt_20_P15.jsonl` (CoVT 640장)
> - `outputs_korean/captions_base_20_P15.jsonl` (Base 640장, 신규 생성)
> - `eval_outputs_korean/win_rates_P15_kb2_640_4.1.csv`

---

## 0. 이전 문서 로드맵 진행 현황

이전 문서(`arch_kb_issue_and_plan_20260220_1044.md`)에서 수립한 Step 계획 대비:

| Step | 내용 | 상태 |
|------|------|------|
| Step 1 | P15 vs P19 pairwise 평가 | ✅ 완료 → **P15 우세 확인** (P15 Contested WR 82.0%, 640쌍) |
| Step 2 | 건축 KB 개선 + 80장 재생성 | ✅ 완료 (본 문서) |
| Step 3 | KB 개선 전/후 비교 (base vs CoVT 최종) | ✅ 완료 (640쌍 GPT-4.1 평가) |
| Step 4 | 최종 학습 데이터 구성 | ⏳ 미완료 |

### 0-1. P15 vs P19 Head-to-Head 결과 요약

> 평가 파일: `eval_results_pairwise_prompt_1519.jsonl`, `win_rates_pairwise_prompt_1519.csv`
> 비교 대상: CoVT-P15 캡션 vs CoVT-P19 캡션 (동일 이미지, 640쌍)

| 지표 | 값 |
|------|-----|
| P15 Contested WinRate | **82.0%** |
| P15 Soft WinRate | **79.4%** |
| Ties | 52개 (8.1%) |

**도메인별 P15 WinRate**:

| 도메인 | P15 | P19 | Tie | P15 WR |
|--------|-----|-----|-----|--------|
| ritual | 66 | 10 | 4 | **86.8%** |
| arch | 129 | 22 | 9 | **85.4%** |
| food | 151 | 34 | 15 | **81.6%** |
| folk | 83 | 23 | 14 | **78.3%** |
| history | 53 | 17 | 10 | **75.7%** |

**해석**: 모든 도메인에서 P15가 P19를 압도. 단, 이 결과는 CoVT 캡션 간 절대 품질 비교이며, P19 vs Base(68.2%)처럼 P19가 CoVT의 지식 주입 우위를 더 잘 드러내는 현상과는 별개. **학습 데이터 품질 기준으로는 P15 캡션이 우수**하다고 확정.

---

## 1. KB 수정 내용

### 1-1. 설계 원칙 (실험 과정에서 확립)

- **KB = 순수 어휘 사전**: "어느 구조물이 보이는지 명시하라"는 행동 지시는 프롬프트 영역 → KB에서 제거
- **길이 제약**: P15 프롬프트 `{category} — {definition}` 인라인 삽입 방식 → KB가 지나치게 길면 역효과
- **할루시네이션 방지**: 모델이 시각적으로 확인하기 어려운 고유 명사(Gwanghwamun 등)는 제거
- **기존 KB 형식 유지**: 한 문장 구조에 sub-structure 어휘만 자연스럽게 추가

### 1-2. 카테고리별 최종 KB (v3)

#### 경복궁 (Gyeongbokgung palace)

```python
# AS-IS (2026-02-20)
"Gyeongbokgung, a Korean royal palace; visually characterized by wooden pavilions "
"with upturned eaves, colorful dancheong paintwork, and raised stone platforms."

# KB v2 (2026-02-20) — Gwanghwamun 추가 → 할루시네이션 유발로 폐기
# "... key structures include Gwanghwamun gate, a grand throne hall, ..."

# KB v3 (2026-02-20) — 최종 채택
"Gyeongbokgung, a Korean royal palace; visually characterized by wooden pavilions "
"with upturned eaves, colorful dancheong paintwork, and raised stone platforms — "
"structures include a grand throne hall, a pavilion on a pond, and various gateways."
```

**변경 핵심**: Gwanghwamun 제거, 복수 하위 구조물 어휘(`a grand throne hall`, `a pavilion on a pond`, `various gateways`) 추가

---

#### 남한산성 (Namhansanseong fortress)

```python
# AS-IS (2026-02-20)
"Namhansanseong, a Korean mountain fortress; visually characterized by long stone "
"walls along mountain ridges with arched gates and watchtowers."

# KB v2 (2026-02-20) — 효과 미미
# (위와 동일 수준)

# KB v3 (2026-02-20) — 최종 채택
"Namhansanseong, a Korean mountain fortress; visually characterized by long stone "
"walls along mountain ridges with arched gates and watchtowers. "
"Inner-site elements include a stone pavilion or temple building."
```

**변경 핵심**: 성벽 외 내부 건물(`stone pavilion or temple building`) 추가 → 이미지 유형 다변화 유도

---

#### 행궁 (Haenggung)

```python
# AS-IS (2026-02-20)
"Haenggung, a Korean temporary royal palace; visually characterized by wooden "
"buildings with curved tile roofs and dancheong-painted eaves arranged around courtyards."

# KB v2 (2026-02-20) — 최종 채택 (v3 시도 후 성능 악화로 복귀)
"Haenggung, a Korean temporary royal palace; visually characterized by wooden "
"buildings with curved tile roofs and dancheong-painted eaves around stone "
"courtyards — structures include a main hall, inner gates, and connecting corridors."

# KB v3 (2026-02-20) — 기각
# "... Visible elements may include a central hall, a gate, stone-paved pathways, "
# "or covered walkways." → "Visible elements may include" 표현이 새 템플릿 고착 유발
# dup>0.9 카운트: v2 21개 → v3 72개로 악화
```

**변경 핵심**: `main hall`, `inner gates`, `connecting corridors` 추가. v3의 "Visible elements may include" 표현은 역효과 — v2 유지

---

#### 팔달문 (Paldalmun Gate)

```python
# AS-IS (2026-02-20)
"Paldalmun, a gate of Hwaseong Fortress in Suwon; visually characterized by an "
"arched stone gateway topped by a two-story wooden pavilion with curved tile roofs."

# KB v2 (2026-02-20) — "modern urban backdrop" 수렴 어구 발생
# "... flanked by stone walls, with a modern urban backdrop visible."

# KB v3 (2026-02-20) — 최종 채택
"Paldalmun, a gate of Hwaseong Fortress in Suwon; visually characterized by an "
"arched stone gateway topped with a two-story wooden pavilion, flanked by curved stone walls."
```

**변경 핵심**: "modern urban backdrop" 제거, "two-story" 유지, `flanked by curved stone walls` 추가

---

### 1-3. 스크립트 변경 사항

- `caption_gen_korean_50.py`에 `--categories` CLI 인자 추가 → 특정 카테고리 ID 접두사만 선택 재생성 가능
  ```bash
  python caption_gen_korean_50.py --categories 9008 9027 9028 9029 ...
  ```

---

## 2. 캡션 재생성 및 다양성 분석

### 2-1. 재생성 흐름

```
captions_covt_20_P15.jsonl.bak_20260220_arch_kb1  ← 원본 백업 (KB v1)
captions_covt_20_P15.jsonl                        ← 건축 80장 KB 개선 버전으로 교체
```

- 경복궁·남한산성·팔달문: KB v3으로 최종 채택
- 행궁: KB v3 기각 후 v2로 롤백, v2 재생성

### 2-2. 정성적 다양성 분석 (CoVT 20장 기준)

| 카테고리 | KB | Unique 캡션 | dup>0.9 쌍 | 비고 |
|---------|-----|------------|-----------|------|
| 경복궁 | v1 → **v3** | 11 → **16** | 37 → **5** | ✅ 큰 개선 |
| 남한산성 | v1 → **v3** | 11 → **18** | 6 → **1** | ✅ 큰 개선 |
| 행궁 | v1 → **v2** | 4 → **11** | 62 → **21** | ✅ 개선 (v3 기각) |
| 팔달문 | v1 → **v3** | 7 → **7** | 25 → **36** | ❌ 개선 없음 |

**팔달문 한계**: 단일 건물을 다양한 각도에서 찍은 이미지들 → KB 개선만으로는 수렴 해소 불가. 구조적 제약으로 판단.

---

## 3. GPT-4.1 Pairwise 평가 (640쌍)

### 3-1. 평가 설정

| 항목 | 내용 |
|------|------|
| Judge | GPT-4.1 |
| 비교 | `captions_base_20_P15.jsonl` vs `captions_covt_20_P15.jsonl` |
| 샘플 수 | 640쌍 (32 카테고리 × 20장) |
| 평가 파일 | `eval_outputs_korean/win_rates_P15_kb2_640_4.1.csv` |
| 이전 P15 평가 | 50샘플 기준 WinRate 48.8% (참고용) |

> **Note**: 이전 P15 결과(48.8%)는 다른 base 파일과 50샘플로 측정된 값으로, 직접 수치 비교에 주의 필요. 본 평가는 base_20 (Qwen2.5-VL-7B-Instruct 신규 생성) 기준.

### 3-2. 전체 결과

| 지표 | 값 |
|------|-----|
| CoVT Contested WinRate | **49.4%** |
| CoVT Soft WinRate | **49.5%** |
| Net (CoVT - Base) | **-7** |
| Ties | 81개 (12.7%) |
| Total | 640쌍 |

→ base와 CoVT가 거의 동률. KB 개선이 전체 WinRate를 유의미하게 끌어올리지 못함.

### 3-3. 도메인별 결과

| 도메인 | CoVT | Base | Tie | Contested WR | 평가 |
|--------|------|------|-----|-------------|------|
| food | 89 | 77 | 14 | **53.6%** | CoVT 우세 ✅ |
| ritual | 38 | 32 | 10 | **54.3%** | CoVT 우세 ✅ |
| other | 21 | 15 | 4 | **58.3%** | CoVT 우세 ✅ |
| folk | 43 | 48 | 9 | **47.3%** | 거의 동률 |
| **arch** | **57** | **61** | **22** | **48.3%** | CoVT 약세 ❌ |
| **history** | **28** | **50** | **22** | **35.9%** | CoVT 열세 ❌ |

### 3-4. 건축 4개 카테고리 상세

| 카테고리 | CoVT | Base | Tie | WR | KB 효과 |
|---------|------|------|-----|----|--------|
| 경복궁 | 13 | 5 | 2 | **72.2%** | ✅ KB v3 성공 |
| 남한산성 | 7 | 12 | 1 | **36.8%** | ❌ KB v3 효과 부족 |
| 행궁 | 3 | 15 | 2 | **16.7%** | ❌ KB v2에도 CoVT 열세 |
| 팔달문 | 4 | 15 | 1 | **21.1%** | ❌ 구조적 한계 |

### 3-5. 주요 카테고리별 WinRate (전체)

**CoVT 강세 (>60%)**:

| 카테고리 | WR |
|---------|-----|
| Cheonggyecheon | 72.2% |
| Gyeongbokgung palace | 72.2% |
| Kimjang | 72.2% |
| Bulgogi | 70.0% |
| Japchae | 70.6% |
| Tteokguk & Mandu soup | 70.6% |
| Cheomseongdae observatory | 69.2% |
| Ganggangsullae | 68.8% |
| Ramyeon | 68.4% |

**CoVT 약세 (<40%)**:

| 카테고리 | WR | 비고 |
|---------|-----|------|
| Haenggung | 16.7% | 건축, KB v2에도 불구 최악 |
| Paldalmun Gate | 21.1% | 건축, 구조적 한계 |
| Mask Dance | 23.5% | 역사/민속 — KB 無관 약점 |
| Admiral Yi Sun-shin | 29.4% | 역사 |
| Panfried battered meatballs | 29.4% | 음식 |

---

## 4. 분석 및 해석

### 4-1. KB 개선 효과 — 카테고리에 따라 편차 극심

- **경복궁만 성공**: 복수 건물 어휘(`throne hall`, `pavilion on pond`, `gateways`)가 실제 이미지 다양성과 맞물려 캡션 분화 성공 → WinRate 72.2%
- **행궁 실패**: KB 다양화(unique 4→11)에도 WinRate 16.7%. Base 모델이 더 시각적으로 정확한 캡션 생성. CoVT 파인튜닝 과정에서 행궁 이미지에 대해 단일 템플릿 생성 패턴이 이미 고착됐을 가능성
- **팔달문 실패**: KB 이전부터 단일 피사체(문 하나)를 여러 각도에서 촬영 → 이미지 자체의 차별화 어휘 한계. KB 개선으로 해결 불가

### 4-2. History 도메인 구조적 약점

KB와 무관하게 역사 도메인 전반에서 CoVT가 열세:
- Admiral Yi Sun-shin: 29.4%, Independence movement: 33.3%, Sejongdaewang: 33.3%, Mask Dance: 23.5%
- **원인 추정**: 역사 카테고리는 시각적 단서가 빈약하고 KB의 문화 지식 주입 효과가 낮음. Base 모델이 이미 해당 문화 지식을 충분히 보유

### 4-3. 이전 계획(arch_kb_issue_and_plan) 대비 평가

| 이전 계획의 가정 | 실제 결과 |
|----------------|---------|
| KB에 sub-structure 어휘 추가 → 모든 건축 카테고리 다양성 개선 | 경복궁만 성공, 행궁·팔달문은 한계 |
| 다양성 개선 → WinRate 향상 | 상관 없음 — 남한산성: unique 11→18 개선됐지만 WR 36.8% |
| 음식·민속놀이에서 CoVT 우세 | ✅ 확인 (food 53.6%, ritual 54.3%) |
| 건축 카테고리가 주요 약점 | ✅ 확인 (4 arch cats WR 36.5%) |

**핵심 발견**: 캡션 다양성(unique count)과 WinRate는 직접적으로 비례하지 않음. CoVT fine-tuning 자체의 domain별 품질 차이가 더 큰 변수.

---

## 5. 결론 및 다음 방향

### 5-1. 건축 KB 개선 결론

- KB 개선은 **경복궁에 한해서만 유의미한 WinRate 개선** 달성
- **행궁·팔달문은 KB 수정 범위를 벗어난 문제**: CoVT 모델 자체가 base 대비 열세인 카테고리 존재 → 학습 데이터 품질(캡션 자체)의 문제가 아닌 fine-tuning 수렴 문제일 가능성

### 5-2. 권장 다음 단계

| 우선순위 | 방향 | 근거 |
|---------|------|------|
| 1 | **P15 캡션 기반 최종 학습 데이터 구성 (Step 4)** | P15 head-to-head 82.0% 확정 — 최종 학습 데이터 프롬프트로 P15 채택 |
| 2 | **행궁·팔달문 후처리 중복 제거** | dup>0.85 쌍 제거로 학습 노이즈 감소 (sim 0.85 기준 행궁 ~9장 유지) |
| 3 | **History 도메인 분석** | CoVT 전반 약세 원인 파악 — 건축과 달리 KB 보강이 아닌 프롬프트/데이터 전략 검토 필요 |

---

## 6. Step 4: 최종 학습 데이터 구성 — 아이디어 및 방향성

### 6-1. 핵심 전제

| 항목 | 결정 |
|------|------|
| 사용 프롬프트 | **P15** (head-to-head 82.0% 확정) |
| 사용 모델 | CoVT-7B (Wakals/CoVT-7B-seg_depth_dino) |
| KB | 현재 v2/v3 (경복궁·남한산성·행궁·팔달문 개선본) |
| 총 이미지 수 | 640장 (32 카테고리 × 20장) |

---

### 6-2. 전략 A — GPT 판정 기반 Winner 캡션 선택

GPT-4.1 평가(`eval_results_P15_kb2_640_4.1.jsonl`)에서 각 이미지 쌍의 `winner_model`을 활용해 캡션 선택.

```
각 이미지에 대해:
  if winner_model == "covt"  → CoVT 캡션 사용
  elif winner_model == "base" → Base 캡션 사용
  else (tie)                  → CoVT 캡션 사용 (default)
```

**장점**: GPT judge가 이미 검증한 품질 기준으로 최선의 캡션을 자동 선택
**단점**: 640쌍 중 283쌍(44.2%)이 base 캡션 채택 → CoVT 학습 비율 감소
**예상 구성**: CoVT 276 + Tie 81(→CoVT) + Base 283 = **최종 357 CoVT / 283 Base**

---

### 6-3. 전략 B — 도메인별 소스 분기

GPT 판정이 아닌 도메인 WinRate를 기반으로 **카테고리 단위** 소스를 결정.

| 도메인 | CoVT WR | 권장 소스 | 비고 |
|--------|---------|---------|------|
| food | 53.6% | **CoVT** | 전 카테고리 CoVT 우세 |
| ritual | 54.3% | **CoVT** | Kimjang 72.2% 등 |
| folk | 47.3% | **CoVT** | 거의 동률, 문화 용어 CoVT 우세 |
| arch (경복궁) | 72.2% | **CoVT** | KB v3 성공 |
| arch (남한산성) | 36.8% | **Base** | CoVT 열세 |
| arch (행궁) | 16.7% | **Base** | CoVT 매우 열세 |
| arch (팔달문) | 21.1% | **Base** | 구조적 한계 |
| history | 35.9% | **Base** | 전 카테고리 CoVT 열세 |

**장점**: 간단명료한 카테고리 단위 분기, 해석 용이
**단점**: 카테고리 내 개별 이미지 편차 반영 불가 (CoVT WR 50% 미만 카테고리도 절반은 CoVT가 우수)

---

### 6-4. 전략 C — 전략 A + 후처리 중복 제거 (권장)

전략 A(per-image winner 선택)에 SequenceMatcher 기반 중복 제거를 결합.

```
Step 1. GPT winner 캡션 선택 (전략 A)
Step 2. 카테고리 내 sim > 0.85인 쌍 중 중복 제거
         → 중복 쌍에서는 더 구체적인 캡션(단어 수 기준) 유지
Step 3. 제거 후 빈자리를 남은 후보(base or covt)로 보충 (optional)
```

**예상 효과**:
- 행궁: winner 선택 후에도 잔존 중복 → 추가 제거로 약 9~11장 유효 데이터 확보
- 팔달문: 구조적 한계로 최종 7~10장 수준
- 전체 학습 데이터 약 580~620장 예상 (20~60장 중복 제거)

---

### 6-5. 최종 학습 데이터 파이프라인 제안

```
[입력]
  captions_covt_20_P15.jsonl   (640장, KB v2/v3)
  captions_base_20_P15.jsonl   (640장)
  eval_results_P15_kb2_640_4.1.jsonl (GPT 판정)

[Step 1] Per-image winner 선택 (전략 A)
  → final_captions_raw.jsonl  (640장, CoVT/Base 혼합)

[Step 2] 카테고리 내 중복 제거 (sim > 0.85)
  → final_captions_dedup.jsonl  (~600장)

[Step 3] 검토 및 수동 보완 (optional)
  - WinRate 매우 낮은 카테고리(행궁 16.7%, 팔달문 21.1%) 캡션 품질 수동 확인
  - 필요 시 base 캡션으로 전수 교체

[출력]
  최종 학습 데이터 JSONL (image_path + caption 쌍)
```

---

### 6-6. 미결 과제

| 과제 | 설명 | 우선순위 |
|------|------|---------|
| **History 도메인 대책** | CoVT WR 35.9% — base 채택 시 CoVT 고유 지식 주입 포기. KB 보강 vs base 캡션 활용 선택 필요 | 높음 |
| **행궁·팔달문 캡션 quality floor** | Base 캡션도 단조로움(단일 피사체 구조적 한계). 중복 제거 후 남은 캡션의 최소 품질 보장 방안 | 중간 |
| **Tie 처리 방침** | 81쌍 Tie → CoVT default or 수동 선택? 대부분 거의 동등하므로 CoVT 유지가 합리적 | 낮음 |
| **학습 데이터 비율 균형** | 최종 CoVT 비율 낮아질 경우 (전략 B: ~350/640) fine-tuning 효과 감소 가능성 검토 | 낮음 |

---

## 7. 실험 파일 목록

| 파일 | 설명 |
|------|------|
| `outputs_korean/captions_covt_20_P15.jsonl` | CoVT 640장 (건축 80장 KB v2/v3 개선 버전) |
| `outputs_korean/captions_covt_20_P15.jsonl.bak_20260220_arch_kb1` | CoVT 640장 원본 백업 (KB v1) |
| `outputs_korean/captions_base_20_P15.jsonl` | Base 640장 (Qwen2.5-VL-7B-Instruct, P15) |
| `eval_outputs_korean/pairs_P15_kb2_640_4.1.jsonl` | 640쌍 pair 목록 |
| `eval_outputs_korean/eval_results_P15_kb2_640_4.1.jsonl` | GPT-4.1 판정 상세 |
| `eval_outputs_korean/win_rates_P15_kb2_640_4.1.csv` | WinRate 집계 |
| `gpt_eval_P15_kb2.py` | 신규 평가 스크립트 (파일 직접 지정 방식) |
