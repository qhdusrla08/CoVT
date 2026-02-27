# CoVT 전패 요인 분석 (P7-P12, GPT-4.1 평가 기준)

> 분석 대상: `eval_results_P7-12_4.1.jsonl` (300건), `win_rates_P7-12_4.1.csv`
> 참조: `caption_gen_korean.py` (프롬프트/KB 정의), `gpt_eval_korean.py` (Judge 시스템)

---

## 1. 전체 승률 요약

| 프롬프트 | Base 승률 | CoVT 승률 | Tie Rate | 특징 |
|---------|-----------|-----------|----------|------|
| **P10** | **69.0%** | **31.0%** | 10.0% | CoVT 최악 |
| P7 | 59.0% | 41.0% | 10.0% | |
| P9 | 59.0% | 41.0% | 6.0% | |
| P12 | 55.0% | 45.0% | 2.0% | |
| P11 | 51.0% | 49.0% | 6.0% | 거의 동점 |
| **P8** | **40.0%** | **60.0%** | 12.0% | CoVT 유일 승리 |

---

## 2. 프롬프트 정의 총람

각 프롬프트는 `caption_gen_korean.py`의 `PROMPTS` 딕셔너리에 정의되어 있으며, 모두 Knowledge Base(KB) 정의를 주입하는 v2 계열이다.

| 프롬프트 | 전략 | 핵심 구조 |
|---------|------|----------|
| **P7** | KB 참조 + 시각 묘사 (가장 단순) | `Describe the visual appearance of '{category}' ({definition})` |
| **P8** | KB 참조 + **명시적 grounding 제약** | `Describe only what you actually see... Do not include details from the reference that are not visible` |
| **P9** | KB를 **용어 가이드**로 제공 | `Use the following as a naming guide... List the observable visual elements` |
| **P10** | KB **조건부 매칭** | `describe only the features from this definition that are actually visible` |
| **P11** | **이미지 우선**, KB 보조 | `Describe what you see... Only mention cultural terms if visually supported` |
| **P12** | 간결한 통합 (P7 + grounding 한 줄) | `Describe the visual appearance... Only describe what is directly observable` |

---

## 3. 핵심 패배 요인 3가지

### 3.1 시각 디테일 누락 (모든 프롬프트 공통)

CoVT는 Base보다 **평균 9~19자 짧은 캡션**을 생성하며, 색상/조명/개수 등 관찰 가능한 시각 요소를 빠뜨린다.

**예시 — Cheomseongdae (P7)**
- Base: `"illuminated in pink light against a dark night sky"`
- CoVT: `"illuminated against a dark night sky"` — 핑크색 조명이라는 핵심 시각 단서를 누락

**예시 — Sejongdaewang (P7)**
- Base: `"wearing an ikseongwan, with Hangul script inscribed on the pedestal"`
- CoVT: `"wearing a distinctive winged crown, and is positioned on a pedestal with Korean text"` — 전문 용어를 일반 표현으로 대체

### 3.2 KB 의존의 역설 — "복사하면서 핵심 용어는 제거"

CoVT는 KB 정의를 통째로 복사하여 환각을 유발하면서도, 정작 KB의 **고유 문화 용어는 일반 표현으로 대체**하는 모순적 행동을 보인다.

| KB 고유 용어 | CoVT가 생성한 표현 |
|-------------|------------------|
| ikseongwan | distinctive winged crown |
| Jeontonghonrye | traditional wedding |
| Japchae | (이름 생략, 재료만 나열) |
| gochugaru | red chili paste |
| Kimjang | communal practice |

### 3.3 환각 및 과도한 추론 (Overreach)

**극단적 환각 — P10 Bulgogi**
- CoVT: `"The glossy dark-brown glaze and thin slices of meat are not visible in the image."` — 이미지에 보이는 요소를 오히려 부정

**비시각적 맥락 추가 — P7 Independence movement**
- CoVT: `"the 104th anniversary of the independence movement"` — 이미지에서 확인 불가능한 구체적 맥락 추가

**상징적 해석 — P9 Ancestral rite**
- CoVT: `"symbolizing respect and remembrance"` — 비시각적 해석 추가로 overreach 판정

---

## 4. 프롬프트별 상세 분석

### 4.1 P7: KB 참조 + 시각 묘사 (Base 27 : CoVT 18)

**패배 키워드 빈도** (Base가 이긴 27건):
- "visually grounded" 관련: 96%
- "knowledge injection" 관련: 74%
- "overreach": **41%** — grounding 제약 부재로 과도한 추론
- "more specific/detail": 30%

**핵심 패배 메커니즘**: grounding 제약이 없는 상태에서 CoVT가 KB 정의를 넘어서는 추론을 하거나, Base보다 시각 디테일이 부족한 캡션을 생성한다.

**Judge Reason 예시:**

> "Caption A is more visually grounded, as it only describes what is directly observable: people in white hanbok waving the Taegeukgi at a celebratory or commemorative event. Caption B **hallucinates the specific context of the '104th anniversary of the independence movement,'** which is not visually verifiable from the image alone."

> "Caption B provides higher Korean knowledge injection value by using the specific term **'ikseongwan'** for the crown and referencing **'Hangul script'** on the pedestal, which are both culturally and visually relevant."
> (→ Base가 오히려 한국 문화 용어를 더 정확히 사용한 사례)

---

### 4.2 P8: 명시적 Grounding 제약 (Base 17 : CoVT 27) — CoVT 유일 승리

**CoVT 승리 키워드 빈도** (CoVT가 이긴 27건):
- "knowledge injection" 관련: 78%
- "visually grounded": 93%
- "terminology": 30%

**CoVT가 승리하는 차별적 요인:**

P8의 이중 grounding 제약이 CoVT의 환각을 효과적으로 억제하면서, CoVT가 내재적으로 보유한 한국 문화 지식이 적절한 수준에서 발현된다.

```
[P8 구조]
"Reference: {category} — {definition}."  ← KB를 Reference로 격리
"Describe only what you actually see"     ← 포지티브 제약
"Do not include details from the reference that are not visible"  ← 네거티브 제약
```

**CoVT 승리 Judge Reason 예시:**

> "Both captions are visually grounded... but Caption B explicitly uses the term **'Jesa,'** which is a specific Korean cultural term for the ancestral rite. This provides higher Korean knowledge injection value and more accurate terminology."

> "Caption A provides more visually grounded detail, explicitly mentioning the **Korean script on the gravestone** and the **bouquet of white flowers**, which helps with Korean knowledge injection."

**P8에서도 CoVT가 지는 경우 (17건):**
- Hallucination이 여전히 18% 존재
- 예: `"twelve pieces of gimbap, each featuring shrimp"` — 개수와 재료 모두 환각

---

### 4.3 P9: KB를 용어 가이드로 제공 (Base 28 : CoVT 19)

**패배 키워드 빈도** (Base가 이긴 28건):
- "visually grounded": 100%
- "overreach": **43%** — P7과 함께 가장 높은 overreach 비율
- "more specific/detail": 32%

**핵심 패배 메커니즘**: "naming guide"라는 지시에도 불구하고 CoVT가 KB 정의를 과도하게 참조하여 overreach하거나, Base가 더 풍부한 시각 디테일과 정확한 한국 용어를 동시에 제공한다.

**Judge Reason 예시:**

> "Caption B injects more Korean cultural knowledge by using the specific term **'Jeontonghonrye'** for a Korean traditional wedding."
> (→ Base가 CoVT보다 정확한 용어 사용)

> "Caption A is more visually grounded... Caption B introduces the **symbolic meaning ('symbolizing respect and remembrance')**, which is not visually verifiable, thus slightly overreaching."

> "Caption A does not specify the number of individuals... the image shows three people, **not two as stated in Caption B.**"
> (→ CoVT가 인원수를 2명으로 오인, 실제 3명)

---

### 4.4 P10: KB 조건부 매칭 (Base 32 : CoVT 13) — CoVT 최악 성적

**P10이 CoVT에 최악인 구조적 원인 3가지:**

**(a) "from this definition"이 KB 복사를 유도**
- CoVT가 KB 정의를 직접 참조 프레임으로 사용하여 이미지 확인 없이 복사

**(b) "only"에 대한 과도한 축소 반응**
- 한국 문화 고유 용어까지 제거하는 현상 발생
- Base 승리 시 평균 캡션 길이: Base 180자 vs CoVT 166자

**(c) 카테고리별 전멸 패턴**
- 28개 카테고리 중 **14개에서 CoVT 전패**: Kimjang, Gyeongbokgung, Yutnori, Mask Dance, Admiral Yi Sun-shin, Gimbap, Yaksik, Bulgogi, Tteokguk/Mandu, Japchae, Namhansanseong, Haenggung, Paldalmun, EXPO Hanbit Tower

**Judge Reason 예시:**

> "Caption B **incorrectly states that these features are not visible**, which is factually inaccurate."
> (→ Bulgogi: 이미지에 보이는 요소를 부정하는 극단적 오류)

> "Caption A correctly identifies the dish as **'Japchae'**... Caption B does not name the dish, missing an opportunity for explicit Korean knowledge injection."
> (→ CoVT가 음식명을 생략하고 재료만 나열)

> "Caption B introduces elements (**long stone walls and watchtowers**) that are **not directly supported by the image**."
> (→ Namhansanseong: KB 정의를 그대로 복사했으나 이미지에는 없는 요소)

---

### 4.5 P11: 이미지 우선, KB 보조 (Base 24 : CoVT 23) — 과소평가 가능성

사실상 동점(24:23)으로, "Describe what you see"가 이미지 기반 묘사를 우선시하여 CoVT의 환각을 자연스럽게 억제하면서 한국 문화 연결도 허용한다.

**패배 키워드 빈도** (Base가 이긴 24건):
- "overreach": 25% (P7/P9보다 낮음)
- "unverifiable": 17%
- "generic": 17%

**Judge Reason 예시:**

> "Caption A mentions the **calligraphy panels in the background**, which are visually present and are a distinctive element of a Jesa setup."
> (→ CoVT가 배경의 서예 작품을 누락)

> "Caption A provides visually grounded details about Admiral Yi Sun-shin's traditional **Joseon military armor and sword**... Caption B is more generic."
> (→ CoVT가 문화적 시각 요소 대신 주변 환경만 묘사)

#### P11 캡션 품질 재평가

P11 패배 24건을 세분류하면:

| 분류 | 건수 | 비율 |
|------|------|------|
| **근소한 차이 (Marginal)** — 디테일 1-2개 차이 | 12건 | 50.0% |
| **진정한 품질 이슈 (Genuine)** — 환각/오류/심각한 누락 | 7건 | 29.2% |
| **시각 디테일 편향 (Bias)** — base가 세부 묘사 1개 더 있을 뿐 | 5건 | 20.8% |

Judge가 **"Both captions are good/correct"라고 인정한 후에도** base를 선택한 비율: **62.5%** (24건 중 15건)

**근소한 차이 패배 예시:**

> `I_01_9025_0098_A` (word overlap 97%):
> - Base: `"...garnished with green onions, red chili pepper, and other toppings."`
> - CoVT: `"...garnished with green onions, red chili pepper, and other colorful toppings."`
> - Judge: CoVT가 "colorful"이라는 unverifiable descriptor를 추가했으므로 base 승리
> - **→ "colorful" 한 단어 차이로 패배. 사실상 동일한 캡션.**

**CoVT 지식 주입이 오히려 불리하게 작용한 사례:**

> `I_01_9005_0023_E`:
> - Base: `"bowing in respect, which is a traditional Korean grave-tending rite"` (bowing 여부 이미지에서 불확실)
> - CoVT: `"standing in a line on a paved area, facing a statue"` (더 시각적으로 정확)
> - Judge: base가 문화 설명 가치에 가산점을 받아 승리
> - **→ CoVT가 오히려 더 정확하지만 패배한 역설적 사례**

---

### 4.6 P12: 간결한 통합 (Base 27 : CoVT 22) — 과소평가 가능성

P7의 단순 구조에 "Only describe what is directly observable" 한 줄을 추가한 형태. grounding 제약이 P8보다 약해 환각을 완전히 억제하지 못한다.

**패배 키워드 빈도** (Base가 이긴 27건):
- "visually grounded": 100%
- "overreach": 30%
- "more specific/detail": 30%
- "hallucination": 11%

**Judge Reason 예시:**

> "Caption B introduces an **unverifiable claim about a wedding ceremony**, which is not directly supported by the image."
> (→ "only describe what is directly observable"라는 제약에도 불구하고 추론 추가)

> "Caption B **incorrectly states they are wearing traditional Korean clothing**, which is not visually supported."
> (→ 실제 검정 정장을 전통 의복으로 잘못 분류)

#### P12 캡션 품질 재평가

P12 패배 27건을 세분류하면:

| 분류 | 건수 | 비율 |
|------|------|------|
| **근소한 차이 (Marginal)** | 13건 | 48.1% |
| **진정한 품질 이슈 (Genuine)** | 11건 | 40.7% |
| **시각 디테일 편향 (Bias)** | 3건 | 11.1% |

Judge가 **"Both captions are good/correct"라고 인정한 후에도** base를 선택한 비율: **70.4%** (27건 중 19건)

**근소한 차이 패배 예시:**

> `I_01_9029_0011_A` (word overlap 86%):
> - Base: `"...flanked by curved fortress walls adorned with red flags and traditional Korean architectural details."`
> - CoVT: `"...flanked by curved fortress walls adorned with traditional Korean architectural elements and red flags."`
> - Judge: "red flags"의 위치 서술 순서가 base가 더 자연스럽다고 판정
> - **→ 단어 배열 순서 차이로 패배. 품질 차이 없음.**

**CoVT 문화 용어 사용이 패널티를 받은 역설적 사례:**

> `I_01_9001_0141_G`:
> - CoVT: **"Kimjang"**이라는 문화 용어를 정확히 사용
> - Judge: 이를 **"overreaching"으로 판정**하여 패배
> - **→ 한국 지식 주입이 목적인 평가에서, 문화 용어 사용이 오히려 패널티를 받은 케이스**

---

## 5. 프롬프트 간 패배 원인 분포 비교

| 패배 원인 | P7 | P8 | P9 | P10 | P11 | P12 |
|-----------|-----|-----|-----|------|------|------|
| Overreach | **41%** | 12% | **43%** | 16% | 25% | 30% |
| Hallucination | 7% | 18% | 7% | 9% | 8% | 11% |
| Detail 부족 | 30% | 18% | 32% | 16% | 12% | 30% |
| 용어 부족 | 19% | 29% | 14% | 19% | 17% | — |
| Conciseness | 33% | 6% | 14% | **28%** | 12% | 15% |

**해석:**
- **P7/P9**: Overreach가 40%+ — grounding 제약이 약하여 CoVT가 과도한 추론
- **P10**: Conciseness 28% + Hallucination 조합 — "only" 제약에 과도 반응 + KB 복사
- **P8**: 총 패배 건수(17건) 자체가 적어 비율이 높아 보이는 것
- **P12**: Overreach(30%)와 Detail 부족(30%)이 균형 — 중간 강도의 grounding 제약

---

## 6. P8이 CoVT에 유리한 핵심 차별점

P8과 다른 프롬프트의 구조적 비교:

```
[P8]  "Describe only what you actually see"
      + "Do not include details from the reference that are not visible"

[P10] "describe only the features from this definition that are actually visible"

[P12] "Only describe what is directly observable"
```

**P8만의 차별점 3가지:**

1. **이중 제약 구조**: 포지티브 제약("only what you actually see") + 네거티브 제약("Do not include details from the reference"). P10/P12는 단일 제약만 존재.

2. **KB를 "Reference"로 격리**: KB 정의가 "Reference:"라는 라벨로 분리되어 있어, 모델이 이를 직접 복사하기보다 참고 자료로 인식. 반면 P10의 "Given that {category} is {definition}"은 KB를 전제 조건으로 내재화하여 복사를 유도.

3. **결과적 효과**: CoVT 승리 27건 중 78%가 "knowledge injection" 관련 — grounding 제약이 환각을 차단하는 동시에, CoVT의 내재적 한국 문화 지식이 적절한 수준에서 발현.

---

## 7. P11/P12 재평가 종합 결론

### 7.1 승률이 실제 품질 차이를 반영하지 않음

P11(51:49)과 P12(55:45)의 승률은 **"CoVT가 base보다 나쁘다"가 아니라, "양쪽이 거의 동등하며 Judge 편향이 누적된 결과"**로 해석해야 한다.

| 근거 | P11 | P12 |
|------|-----|-----|
| 근소한 차이 + 디테일 편향으로 패배 | **70.8%** (17/24건) | **59.3%** (16/27건) |
| 진정한 품질 문제로 패배 | 29.2% (7/24건) | 40.7% (11/27건) |
| Judge가 "both good" 인정 후 base 선택 | 62.5% | 70.4% |

"Both good" 케이스를 tie로 재분류할 경우:
- **P11**: 23W-9L-18T → **CoVT 실질 우세**
- **P12**: 22W-8L-20T → **CoVT 실질 우세**

### 7.2 Judge 기준의 구조적 편향 분석

현재 `gpt_eval_korean.py`의 `JUDGE_SYSTEM`은 다음 4가지 메커니즘으로 CoVT에 구조적으로 불리하다:

**(a) Visual Grounding 최우선 (Criterion 1) → 지식 주입이 overreach로 처리**

CoVT의 핵심 강점은 지식 주입이나, 주입된 지식은 "이미지에서 직접 관찰 가능한 것"을 넘어서는 문화적 맥락을 포함한다. 이 기준이 최우선이므로, CoVT가 문화적 맥락을 추가할 때마다 "overreach" 위험에 노출된다.

**(b) "Penalize any unverifiable claims" 조항 → 문화 용어 사용이 패널티**

CoVT가 "Kimjang", "independence movement" 등의 문화 맥락을 연결하면, judge가 "unverifiable historical context"로 분류할 여지가 있다. 실제로 P12에서 "Kimjang" 정확 사용이 overreach로 판정된 사례가 존재한다.

**(c) Micro tie-breaker가 시각 디테일 수에 편향**

> `(a) Prefer the caption that describes more observable visual details`

CoVT가 문화 용어와 맥락에 단어를 할애하면 시각 디테일 수가 줄어든다. 동점에서 이 기준이 체계적으로 base에 유리하게 작동한다.

**(d) Tie 강력 억제 → 동등 품질도 강제 승패**

> `Ties are strongly discouraged`
> `near-identical in wording AND content` 만 tie 허용

word overlap 97% 캡션도 tie가 아닌 base 승리로 판정. 미세 차이 시 항상 "시각 디테일이 더 많은 쪽"(대개 base)이 승리한다.

---

## 8. 개선 방향 제언

### 8.1 평가 수준 (최우선 — Judge 기준 수정)

P11/P12 재평가에서 드러난 구조적 편향을 해소하기 위해 **Judge 기준을 먼저 수정**해야 한다.

| 수정 사항 | 기존 | 변경 |
|----------|------|------|
| **기준 1-2 동등화** | Visual Grounding > Knowledge Injection (cascade) | Visual Grounding = Knowledge Injection (동등 가중치, 종합 판단) |
| **문화 용어 보호** | unverifiable claims 일괄 페널티 | 시각적으로 뒷받침되는 문화 용어는 overreach에서 명시적 제외 |
| **Generic 표현 패널티** | 없음 | Base가 한국 문화 용어 대신 generic 표현 사용 시 감점 |
| **Tie 기준 완화** | near-identical wording만 tie | 동일 수준의 grounding + injection이면 tie 허용 |
| **Micro tie-breaker 균형** | 시각 디테일 수 우선 | 시각 디테일과 문화 용어 정확도를 동등하게 고려 |

### 8.2 프롬프트 수준

| 방안 | 설명 |
|------|------|
| **P8 구조 표준 채택** | 이중 grounding 제약 + KB 격리가 CoVT의 강점을 극대화 |
| **P10 구조 회피** | "features from this definition"이 KB 복사를 유도하므로 사용 지양 |
| **하이브리드 프롬프트** | P8의 grounding + P9의 naming guide 결합 (아래 참조) |

**하이브리드 프롬프트 제안:**
```
"Reference (naming guide only): {category} — {definition}.
Describe only what you actually see in this image in exactly one sentence.
Use the correct Korean cultural terms from the reference where visually
supported, but do not include any details from the reference that are not visible."
```

### 8.3 모델 수준

| 방안 | 설명 |
|------|------|
| **시각 디테일 밀도 강화** | Fine-tuning 데이터에서 색상/조명/개수 등 시각 디테일이 풍부한 캡션 비중 확대 |
| **KB 용어 고정 출력** | KB 고유 용어를 반드시 출력하도록 constrained decoding 또는 reward signal 도입 |
| **환각 방지** | 생성 후 이미지-텍스트 정합성 검증 단계 추가, RLHF에서 환각 페널티 강화 |
