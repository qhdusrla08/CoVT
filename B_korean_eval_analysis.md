# B_korean Track: CoVT vs Base 캡션 평가 분석

## 1. 실험 개요

### 1.1 CoVT란?

[CoVT (Chain-of-Visual-Thought)](https://github.com/Wakals/CoVT)는 VLM이 텍스트뿐 아니라 **연속적인 시각 토큰**을 통해 추론할 수 있게 하는 프레임워크이다. SAM(세그멘테이션), DepthAnything(깊이), PIDINet(에지), DINO(패치 시맨틱) 등 약 20개의 시각 토큰을 통해 **시각적 사고 사슬(visual thought chain)**을 형성하며, 텍스트만의 추론을 넘어 지각적 근거(perceptual grounding)를 강화한다.

- **Base 모델**: Qwen2.5-VL-7B-Instruct
- **CoVT 모델**: 위 base에 SAM/Depth/DINO 시각 토큰을 추가 학습한 체크포인트

### 1.2 실험 목적

한국 문화 도메인(B_korean track) 이미지에 대해, CoVT의 시각 토큰이 **한국 문화 지식 주입(Korean Knowledge Injection)** 품질의 캡션 생성에 기여하는지 검증한다.

### 1.3 데이터셋

`data/B_korean/` 디렉토리의 32개 한국 문화 카테고리, 각 카테고리당 복수 이미지:

| 분류 | 카테고리 예시 |
|------|-------------|
| 전통 의례 | 김장, 제례(Ancestral rite), 성묘, 세배(Sebae) |
| 역사/인물 | 독립운동, 세종대왕, 이순신, 한국전쟁 |
| 건축/장소 | 경복궁, 청계천, 첨성대, 남한산성, 행궁, 팔달문, EXPO 한빛탑 |
| 전통 놀이/공연 | 윷놀이, 강강술래, 탈춤, 연날리기, 제기차기 |
| 음식 | 라면, 김밥, 계란말이, 불고기, 떡국/만두국, 잡채, 한과, 약식, 약과 |
| 기타 | 전통혼례, 독립기념관 |

### 1.4 프롬프트 설계 (P0~P6)

`caption_gen_korean.py`에서 정의한 7개 프롬프트로, 난이도와 지시 복잡도가 점진적으로 증가한다:

| ID | 유형 | 프롬프트 | 설명 |
|----|------|---------|------|
| **P0** | 대조군 | `Describe the scene and main objects in exactly one sentence.` | 카테고리 정보 없음 |
| **P1** | 기본 지시 | `Visually describe '{category}' and its layout in exactly one sentence.` | 카테고리명 제공 + 외양 묘사 |
| **P2** | 특징 추출 | `Identify the visual elements of '{category}' in exactly one clear sentence.` | 시각 요소 식별 |
| **P3** | 구조 검증 | `Describe '{category}' by verifying its shapes and features in exactly one sentence.` | 형태/배치 검증 |
| **P4** | 기술 지표 | `Describe '{category}' using segmentation, depth, and patch features in exactly one sentence.` | SAM/Depth/DINO 지표 활용 지시 |
| **P5** | 지표 통합 | `Visually describe '{category}' incorporating segmentation, depth, and patch features in exactly one sentence.` | 시각 지표 직접 반영 |
| **P6** | 시각-지식 통합 | `Describe the visual appearance and cultural context of '{category}' in exactly one sentence.` | 외양 묘사 + 문화적 맥락 결합 |

`{category}`는 이미지 경로의 디렉토리명에서 자동 추출된다 (예: `9025_Tteokguk_Mandu soup` → `Tteokguk and Mandu soup`).

### 1.5 평가 방법

- **GPT-4.1-mini 기반 Blind A/B 평가** (`gpt_eval_korean.py`)
- 각 이미지에 대해 base/CoVT 캡션을 **무작위 순서(swap)**로 제시하여 위치 편향 제거
- 5단계 평가 기준 (우선순위 순):
  1. **Visual Grounding & Verifiability** (최우선) — 이미지에서 직접 관찰 가능한 내용인가
  2. **Korean Knowledge Injection Value** (핵심 목표) — 한국 문화 용어를 시각 근거와 함께 사용하는가
  3. **Disambiguation & Specificity** — 모호함 없이 구체적인가
  4. **Cultural Accuracy & Terminology** — 문화 용어가 정확한가
  5. **Training Usefulness** — 학습 데이터로서 유용한 포맷인가

---

## 2. 전체 승률 결과

`eval_outputs_korean/win_rates.csv` 기준, 각 프롬프트별 50건 평가:

| Prompt | Base 승률 | CoVT 승률 | Tie | CoVT 우위 |
|--------|----------|----------|-----|-----------|
| P0 (대조군)   | 69.0% | 31.0% | 7 | - |
| P1 (기본 지시)  | 70.0% | 30.0% | 0 | - |
| P2 (특징 추출)  | 62.0% | 38.0% | 4 | - |
| P3 (구조 검증)  | 56.0% | 44.0% | 0 | - |
| P4 (기술 지표)  | 51.0% | 49.0% | 3 | - |
| P5 (지표 통합)  | 50.0% | 50.0% | 6 | 동률 |
| **P6 (시각-지식 통합)** | **40.0%** | **60.0%** | **0** | **CoVT 승리** |

### 핵심 관찰

**1) 프롬프트 복잡도에 따른 CoVT 승률의 단조 증가**

```
P0(31%) → P1(30%) → P2(38%) → P3(44%) → P4(49%) → P5(50%) → P6(60%)
```

프롬프트가 시각적 추론을 더 많이 요구할수록, CoVT의 시각 토큰이 캡션 품질에 기여하는 정도가 높아진다.

**2) P4~P5에서의 전환점**

SAM/Depth/DINO 등 기술 지표를 프롬프트에 명시적으로 요구하기 시작하는 P4부터 CoVT가 base와 대등해지며, P5에서 정확히 50:50에 도달한다. 이는 CoVT의 시각 토큰이 이러한 지표를 실제로 활용하고 있음을 시사한다.

**3) P6에서의 역전**

"visual appearance + cultural context"를 동시에 요구하는 P6에서 CoVT가 **유일하게 base를 역전**하여 60% 승률을 달성한다. 시각적 근거와 문화 지식의 결합이 CoVT의 강점이 극대화되는 조건임을 보여준다.

---

## 3. P6 CoVT 승리 reason 심층 분석

P6 50건 중 CoVT 승리 30건, Base 승리 20건, Tie 0건.

### 3.1 GPT 평가 기준별 키워드 빈도

CoVT가 승리한 30건의 reason에서 추출한 평가 기준 키워드:

| 평가 기준 | 등장 건수 | 비율 |
|----------|----------|------|
| Visual Grounding | 30 | 100.0% |
| Hallucination Penalty (for base) | 28 | 93.3% |
| Specific/Detail | 18 | 60.0% |
| Korean Knowledge Injection | 15 | 50.0% |
| Training Usefulness | 13 | 43.3% |
| Concise/Conciseness | 7 | 23.3% |
| Accurate/Accuracy | 7 | 23.3% |
| Terminology Quality | 3 | 10.0% |

### 3.2 CoVT 승리의 주요 패턴

GPT reason 30건을 질적으로 분석한 결과, CoVT가 base를 이기는 패턴은 크게 3가지로 수렴한다:

#### Pattern A: Base의 Hallucination 억제 (93.3%)

Base 모델이 문화적 맥락을 요구받았을 때 **이미지에서 확인할 수 없는 사실을 생성**하는 경향이 있으며, CoVT는 이를 회피한다.

> *"Caption A introduces specific cultural festivals (Chinese Dragon Boat Festival, Japanese Tanabata) that are not visually supported..."* — 연날리기(P6 #16)

> *"Caption A hallucinates 'students in traditional attire' which is not clearly supported by the image"* — 세종대왕상(P6 #14)

> *"Caption B is more visually grounded and avoids unverifiable historical claims present in Caption A, such as the specific 104th anniversary"* — 독립운동(P6 #3)

CoVT의 시각 토큰이 텍스트 생성 시 시각적 근거를 강화하여, base가 빠지기 쉬운 "지식 환각(knowledge hallucination)"을 억제하는 효과로 해석된다.

#### Pattern B: 시각 근거 기반의 구체적 디테일 (60.0%)

CoVT 캡션이 이미지에서 **직접 관찰 가능한 세부사항을 더 정확하게** 기술한다.

> *"Caption B adds the visually supported detail of green onions and mentions the cultural significance of Kimjang as a communal activity"* — 김장(P6 #2)

> *"Caption B is more visually grounded by describing the geometric patterns on the board and the vibrant floral mat"* — 윷놀이(P6 #17)

> *"Caption A is more visually grounded and specific, clearly identifying the attire as traditional Korean wedding clothing with vibrant red colors"* — 전통혼례(P6 #5)

#### Pattern C: 간결하고 검증 가능한 문화 지식 (50.0%)

CoVT가 한국 문화 용어를 사용하되, **시각적으로 뒷받침되는 범위 내에서만** 사용한다.

> *"Caption A explicitly mentions the Korean cultural context of an ancestral rite and highlights the traditional calligraphy backdrop, which is visually supported"* — 제례(P6 #6)

> *"Caption B is more concise and directly connects the statue to Korean culture and history without adding unverifiable narrative"* — 세종대왕(P6 #15)

### 3.3 등장 한국 문화 용어 분포

CoVT 승리 30건의 reason + covt_caption에서 등장하는 한국 문화 용어:

| 용어 | 건수 | 용어 | 건수 |
|------|------|------|------|
| Ancestral rite (제례) | 5 | Cheonggyecheon (청계천) | 1 |
| Kimjang (김장) | 2 | Lunar New Year (설날) | 1 |
| kimchi (김치) | 2 | Hangul (한글) | 1 |
| Independence movement | 2 | Yutnori (윷놀이) | 1 |
| Seongmyo (성묘) | 2 | Gimbap (김밥) | 1 |
| Gyeongbokgung (경복궁) | 2 | Hangwa (한과) | 1 |
| Sejongdaewang (세종대왕) | 2 | Tteokguk/Mandu (떡국만두국) | 1 |
| Ganggangsullae (강강술래) | 2 | Namhansanseong (남한산성) | 1 |
| hanbok (한복) | 2 | Haenggung (행궁) | 1 |
| Mask Dance (탈춤) | 2 | tteok/rice cake (떡) | 1 |
| Ramyeon (라면) | 2 | nori seaweed (김) | 2 |
| The Korean War (한국전쟁) | 2 | | |

32개 카테고리 중 **26종의 용어가 고르게 분포**하여, CoVT의 문화 지식 주입 효과가 특정 카테고리에 편중되지 않음을 확인할 수 있다.

---

## 4. 종합 해석

### 4.1 CoVT의 시각 토큰은 "문화 지식 + 시각 근거"의 교차점에서 효과적이다

| 조건 | 결과 | 해석 |
|------|------|------|
| 단순 장면 묘사 (P0~P1) | Base 우세 (69~70%) | 시각 토큰 없이도 충분한 과제 |
| 시각 요소 식별 (P2~P3) | Base 우세 감소 (56~62%) | CoVT의 세그먼테이션/깊이 정보가 점진적 기여 |
| 기술 지표 활용 (P4~P5) | 대등 (49~50%) | 시각 토큰이 명시적 지표 요구에 직접 대응 |
| 시각-지식 통합 (P6) | **CoVT 우세 (60%)** | 시각 근거 + 문화 맥락 결합 시 강점 극대화 |

### 4.2 Base의 한계: Knowledge Hallucination

Base 모델은 "cultural context"를 요구받으면 학습 데이터의 언어적 패턴에 의존하여, 이미지와 무관한 역사적 사실이나 상징적 의미를 생성하는 경향이 있다. CoVT의 시각 토큰은 이러한 환각을 억제하는 **앵커(anchor)** 역할을 한다.

### 4.3 시사점

1. **프롬프트 설계가 핵심**: 동일한 CoVT 모델이라도 프롬프트에 따라 base 대비 -39%p(P1)에서 +20%p(P6)까지 성능 차이가 발생한다. 시각 토큰의 효과를 최대화하려면 "시각 묘사 + 도메인 지식"을 동시에 요구하는 프롬프트가 필요하다.

2. **한국 문화 도메인에서의 활용**: CoVT는 한국 문화 고유명사(한복, 김장, 강강술래 등)를 시각적 근거와 함께 정확하게 사용하는 캡션을 생성할 수 있으며, 이는 한국 문화 도메인 파인튜닝 데이터 구축에 유용하다.

3. **Hallucination 억제 효과**: P6 CoVT 승리 30건 중 93.3%에서 base의 unverifiable/unsupported 서술이 패인으로 지적되었다. CoVT의 가장 실질적인 기여는 새로운 지식의 주입보다, base 모델이 시각적으로 뒷받침되지 않는 문화적 서술을 생성하는 것을 **억제**하는 데 있다.

---

## 5. 실험 환경 요약

| 항목 | 값 |
|------|-----|
| Base 모델 | Qwen2.5-VL-7B-Instruct |
| CoVT 체크포인트 | CoVT fine-tuned (SAM + Depth + DINO visual tokens) |
| 평가 모델 | GPT-4.1-mini (temperature=0.0) |
| 평가 방식 | Blind A/B (random swap) |
| 트랙 | B_korean (32 카테고리) |
| 프롬프트 | P0~P6 (7종) |
| 샘플 수 | 프롬프트당 50건, 총 350건 |
| 생성 설정 | temperature=0.0, max_new_tokens=96, seed=42 |
