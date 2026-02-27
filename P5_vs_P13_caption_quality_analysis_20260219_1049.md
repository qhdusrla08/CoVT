# P5 vs P13 캡션 품질 분석 — Traditional Wedding (9003)

**분석 일시**: 2026-02-19 10:49 KST
**데이터**: `captions_covt_9003_test_P5.jsonl`, `captions_covt_9003_test_P13.jsonl`
**샘플 수**: 각 397개 (카테고리: Traditional wedding 단일)
**모델**: CoVT fine-tuned (`covt_9003_test`)

---

## 1. 프롬프트 구조

| | P5 | P13 |
|---|---|---|
| **템플릿** | `"Visually describe '{category}' incorporating segmentation, depth, and patch features in **exactly one sentence**."` | `"Reference: {category} — {definition}. Describe only what you actually see in this image in **exactly one sentence**, using the correct Korean cultural term where visually supported. Do **not** include details from the reference that are not visible."` |
| **KB 주입** | 없음 | 있음 (Jeontonghonrye, ritual table, mandarin ducks, candles) |
| **Grounding 제약** | 없음 | "only what you actually see" |
| **CoVT 피처 참조** | 있음 (seg/depth/patch) | 없음 |
| **실제 KB 정의 (Traditional wedding)** | — | `"Jeontonghonrye, a Korean traditional wedding; visually characterized by a couple in colorful ceremonial hanbok set before a ritual table, often with wooden mandarin ducks and candles."` |

---

## 2. 기본 통계

| 지표 | P5 | P13 |
|---|---|---|
| 평균 캡션 길이 | 239.7자 | 193.4자 |
| 최소 | 125자 | 102자 |
| 최대 | 396자 | 278자 |
| 1문장 준수 | 100% | 100% |

---

## 3. 할루시네이션 분석

### P5 주요 할루시네이션 유형

#### ① 현대적 요소 혼입 — 17개 (4.3%)

마스크 착용, gymnasium, glass windows, "modern twist" 등 현대 요소를 전통혼례 맥락에서 해석적으로 서술:

```
[0037_E] "...wearing face masks, indicating a modern twist to the traditional ceremony."
[0016_E] "...in front of a modern building with glass windows, creating a striking
           contrast between tradition and contemporary architecture."
[0053_E] "...all wearing masks, indicating a modern adaptation to health protocols."
```

> 일부는 실제 이미지에 현대 요소가 존재할 수 있으나, "modern twist", "health protocols" 등의 해석은 이미지 직접 관찰이 아닌 추론에 해당.

#### ② Generic/정보 없는 상투어 캡션 — 87개 (21.9%)

이미지마다 교체해도 구분이 안 되는 무의미한 묘사. 파인튜닝 데이터로 가장 유해한 유형:

```
[0013_E] "richly detailed with vibrant colors, intricate costumes, and ceremonial
           objects, creating a visually striking tableau."
[0025_E] "...showcasing a blend of tradition and ceremony."
[0141_G] "showcasing the cultural significance of the attire and accessories."
```

색상어조차 없는 완전한 상투어 캡션: **70개 (17.6%)**

#### ③ 추상적·해석적 서술 — 160개 (40.3%)

`symbolizing`, `representing`, `indicating`, `suggesting`, `evoking` 등 해석 동사 과다 사용. 직접 관찰이 아닌 의미 부여.

#### ④ CoVT 피처 용어(`seg/depth/patch`) 실질 효과

`segment` 계열 용어가 캡션에 실제로 등장한 사례: **1건** (일반 명사로 전용). 프롬프트의 의도 대비 캡션 출력에서의 가시적 반영은 제한적이나, CoVT 내부 피처 활용과 직접 대응 여부는 별도 검증 필요.

---

### P13 주요 할루시네이션 유형

#### ① KB 정형 문구 반복 복사 — 106개 (26.7%)

KB 정의 문구(`wooden mandarin ducks and candles`)가 이미지 확인 없이 삽입:

```
[0002_E] "...seated before a ritual table adorned with wooden mandarin ducks and candles."
[0004_E] "...kneeling before a ritual table adorned with wooden mandarin ducks and candles."
[0008_E] "...standing before a ritual table adorned with wooden mandarin ducks and candles."
```

**세부 분류:**

| 유형 | 건수 |
|---|---|
| ritual table과 함께 등장 | 83개 (21.0%) — 일부 이미지 기반 가능 |
| ritual table 없이 단독 등장 | **23개 (5.8%) — 명백한 KB 복사 할루시네이션** |

KB 정의에 `"often with..."` 단서가 있음에도 모델이 조건 없이 삽입함.

#### ② 정형화된 커플 묘사 — 94개 (23.7%)

`"a couple dressed in vibrant hanbok"` 패턴이 94개. 실제로 커플이 아닌 이미지(인형 배치, 차 의식, 신부 단독 등)에도 동일 틀이 적용:

```
[0045_E] P13: "a woman arranging traditional Korean dolls on a table...
               symbolizing the couple's union."
```

#### ③ 현대적 요소 언급 — 8개 (2.0%, P5의 절반)

---

## 4. 캡션 품질 비교

| 지표 | P5 | P13 | 우위 |
|---|---|---|---|
| Jeontonghonrye 사용 | 0개 (0%) | 281개 (70.8%) | **P13** |
| hanbok 사용 | 161개 (40.6%) | 346개 (87.2%) | **P13** |
| ritual table 언급 | 0개 | 147개 (37.0%) | **P13** |
| 동작/행위 묘사 | 200개 (50.4%) | 278개 (70.0%) | **P13** |
| **진짜 시각적으로 풍부** (색상 2+, 구체 물체 2+) | **125개 (31.5%)** | 12개 (3.0%) | **P5** |
| 색상어 2개 이상 | 160개 (40.3%) | 18개 (4.5%) | **P5** |
| Generic/추상적 서술 | 160개 (40.3%) | 85개 (21.4%) | **P13** |
| Generic 상투어 | 87개 (21.9%) | 27개 (6.8%) | **P13** |
| 현대적 요소 혼입 | 17개 (4.3%) | 8개 (2.0%) | **P13** |
| KB 복사 할루시네이션 | 없음 | 23개 명백 | **P5** |
| 1문장 준수 | 100% | 100% | 동등 |

### P5 시각 묘사 품질의 양극화

P5는 best/worst 캡션 분산이 크다:

**Good 사례 (31.5%)**
```
[0010_E] "...a woman in a colorful hanbok seated at a table adorned with offerings,
           a man in a blue robe and a woman in a pink and blue hanbok, all engaged
           in a ceremonial activity under a canopy decorated with red and blue fabric."

[0032_E] "...one person wearing a vibrant red and gold dress adorned with intricate
           patterns, while the other is dressed in a deep purple robe, standing before
           a table laden with ceremonial items such as a vase, a bowl, and a bamboo plant."
```

**Bad 사례 (17.6%)**
```
[0013_E] "richly detailed with vibrant colors, intricate costumes, and ceremonial
           objects, creating a visually striking tableau."
```

---

## 5. P13 Clean 캡션 비율

Jeontonghonrye 있고 mandarin duck 없는 상대적으로 균형 잡힌 캡션:
**198개 (49.9%)**

```
[0006_E] "In the image, a traditional Korean wedding ceremony, known as
           Jeontonghonrye, is taking place outdoors amidst trees, featuring
           a couple dressed in vibrant hanboks standing before a ritual table
           adorned with various ceremonial items."
```

---

## 6. 파인튜닝 데이터 적합성 종합 판단

### P5 — 조건부 사용 가능 (선별 필요)

**결정적 약점:**
- 한국 문화 고유명사 전무 (Jeontonghonrye: 0%). CoVT의 핵심 목적(KB 기반 문화 지식 주입) 미달성
- Generic/추상 묘사 비율 40.3% → 학습 시 상투어 도피 패턴 강화 위험

**보존해야 할 강점:**
- `seg/depth/patch features` 참조 → CoVT 아키텍처(SAM/Depth/DINO 피처)의 능력 활성화 의도. 파인튜닝 데이터에서 제거 불가
- 시각 디테일 풍부한 31.5%는 학습 데이터로 가치 있음
- KB 정의 없으므로 KB 복사 할루시네이션 없음

### P13 — 우선 사용 권장 (단, 필터링 필요)

**강점:**
- 한국 문화 용어 일관 사용 (Jeontonghonrye 70.8%, hanbok 87.2%)
- 동작/행위 묘사 70.0%로 구체적
- Generic 상투어 비율 P5의 1/3 수준

**약점 및 필터링 대상:**
- ritual table 없이 mandarin ducks만 언급하는 23개(5.8%) → 제거 권장
- 동일 고정 문구 반복 94~106개 → 다양성 감소 우려

---

## 7. 개선 방향: P5 카테고리명 변경

### 근거

P5 캡션의 **99.0%(393/397)**가 카테고리명을 캡션에 그대로 반영:

```
157개  "The traditional wedding scene features..."
132개  "The traditional wedding scene is..."
 36개  "A traditional wedding scene featuring..."
```

### 제안

`extract_category_from_path()` 또는 KB 매핑에서 display name을 확장:

```
"Traditional wedding" → "Korean traditional wedding (Jeontonghonrye)"
```

**예상 효과:**
- `"The Korean traditional wedding (Jeontonghonrye) scene features..."` 패턴으로 문화 용어 자동 진입
- KB 정의 없으므로 mandarin ducks 할루시네이션 원천 차단 유지
- `seg/depth/patch features` 지시 그대로 보존 → CoVT 의도 유지

**남는 문제:**
Generic/추상 묘사(~22%)는 카테고리명 변경만으로 해결 안 됨. 이는 프롬프트보다 모델이 이미지에서 구체적 시각 단서를 포착하지 못할 때의 도피 패턴으로, 학습 데이터 선별(상투어 캡션 필터링)이 병행돼야 함.

---

## 8. 결론

| | P5 | P13 |
|---|---|---|
| **파인튜닝 데이터 적합성** | 조건부 (선별 후 사용) | 우선 권장 (필터링 후) |
| **문화 용어** | ❌ 0% → 카테고리명 변경으로 개선 가능 | ✅ 70.8% |
| **시각 디테일** | ✅ 31.5% 풍부 (분산 큼) | ❌ 3.0% |
| **KB 할루시네이션** | ✅ 없음 | ⚠️ 5.8% 명백 |
| **CoVT 피처 참조** | ✅ 보존 | ❌ 없음 |
| **다음 실험 후보** | P5 + category → "Korean traditional wedding (Jeontonghonrye)" | P13 Clean 198개 선별 사용 |

> **최우선 권장**: P19(역대 최고 WinRate 68.2%)를 기반으로 파인튜닝 데이터 생성.
> P5 카테고리명 변경 variant는 KB 주입 없이 CoVT 피처를 활용하는 독립적 실험 경로로 가치 있음.
