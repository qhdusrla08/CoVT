# 건축 카테고리 캡션 유사성 문제 및 KB 개선 계획

> **작성 일시**: 2026-02-20 10:44 KST
> **대상 카테고리**: 경복궁(9008), 남한산성(9027), 행궁(9028), 팔달문(9029)
> **관련 파일**: `caption_gen_korean_50.py`, `captions_covt_20_P15.jsonl`, `captions_covt_20_P17.jsonl`

---

## 1. 문제 확인: 건축 카테고리 캡션 유사성

### 1-1. 측정 결과

| 도메인 | Unique 캡션 비율 | avg 유사도 | pairs > 0.7 |
|--------|----------------|-----------|------------|
| 음식 (P15) | **96.8%** (213/220) | 0.215 | 0.9% |
| 민속놀이 (P15) | **99.0%** (99/100) | 0.263 | 1.0% |
| **건축 (P15)** | **71.2%** (57/80) | **0.327** | **12.0%** |
| **건축 (P17)** | **75.0%** (60/80) | **0.425** | **13.1%** |

**행궁(Haenggung)이 가장 심각**:
- 20개 이미지 → unique 캡션 **9개** (45%만 고유)
- 유사도 > 0.9인 쌍 **62개** (190쌍 중 33%)
- 유사도 > 0.7인 쌍 **160개** (P15 기준)

### 1-2. 카테고리별 세부 현황 (P15)

| 카테고리 | avg 유사도 | max 유사도 | pairs > 0.7 | 심각도 |
|---------|-----------|-----------|------------|--------|
| 9028_Haenggung | 0.765 | 1.000 | 160/190 | ⚠️ 매우 심각 |
| 9029_Paldalmun Gate | 0.683 | 1.000 | 102/190 | ⚠️ 심각 |
| 9008_Gyeongbokgung | 0.575 | 1.000 | 65/190 | 주의 |
| 9027_Namhansanseong | 0.460 | 1.000 | 52/190 | 보통 |

### 1-3. 실제 중복 캡션 사례 (행궁 P15)

아래 9개 이미지가 사실상 동일한 캡션을 가짐:

```
[0009~0015] "The image depicts a traditional Korean Haenggung, featuring
wooden structures with curved tiled roofs and vibrant dancheong-painted
eaves, surrounded by lush greenery under a clear blue sky."
```

이미지별로 촬영 각도, 계절, 관람객 유무 등 차이가 있음에도 동일 캡션 생성.

### 1-4. 크로스 카테고리 유사도 — 건물 간 혼동은 없음

| 비교 | avg 유사도 | pairs > 0.7 |
|-----|-----------|------------|
| 경복궁 vs 행궁 | 0.287 | **0%** |
| 경복궁 vs 남한산성 | 0.285 | **0%** |
| 행궁 vs 팔달문 | 0.186 | **0%** |

→ **서로 다른 건물 간 혼동은 발생하지 않음**. 문제는 동일 건물 내 이미지들끼리의 캡션 수렴.

---

## 2. 근본 원인 분석

### 2-1. 현재 KB의 구조적 한계

현재 건축 KB는 **카테고리 공통 특징**만 기술:

```python
# 현재 Haenggung KB
"Haenggung":
    "Haenggung, a Korean temporary royal palace; visually characterized by
     wooden buildings with curved tile roofs and dancheong-painted eaves
     arranged around courtyards."

# 현재 Gyeongbokgung KB
"Gyeongbokgung palace":
    "Gyeongbokgung, a Korean royal palace; visually characterized by wooden
     pavilions with upturned eaves, colorful dancheong paintwork, and raised
     stone platforms."
```

두 건물 KB가 **거의 동일한 어휘** (wooden / curved / eaves / dancheong)를 공유하며, 이 어휘들은 **모든 이미지에서 동일하게 성립**함.

### 2-2. 음식 vs 건축 KB 본질적 차이

| | 음식 KB | 건축 KB |
|---|---|---|
| 이미지 간 어휘 변동성 | 재료 구성·색상이 이미지마다 다름 | 구조물 자체는 모든 이미지에서 동일 |
| KB 차별화 효과 | 자연스럽게 캡션 다양성 유도 | 모든 이미지에 동일 KB 적용 → 수렴 |
| 이미지별 차별 요소 | 국물 색, 재료 배치, 그릇 형태 등 | 촬영 각도, 계절, 인파, 어느 전각인지 |

### 2-3. 파인튜닝 악영향

1. **이미지-캡션 정렬(visual grounding) 붕괴**: 서로 다른 이미지 10장이 동일 캡션을 가지면 모델이 시각 정보를 무시하고 카테고리명 → 고정 문장 출력 shortcut을 학습
2. **Caption mode collapse**: 건축물 이미지를 보면 무조건 동일 템플릿 문장 생성 (계절·각도·인파 무시)
3. **데이터 비율 왜곡**: 행궁 20장 중 실질 학습 정보는 9개 unique → 동일 패턴의 반복 강화(over-reinforcement)
4. **한국 문화 지식 주입 목적 훼손**: '기능(임시 궁궐)', '위치', '역사적 맥락' 미학습. "dancheong-painted eaves" + "lush greenery"만 반복 강화

---

## 3. KB 개선 방향

### 3-1. 핵심 원칙

현재 KB: **카테고리 공통 정적 특징**만 기술
개선 방향: **공통 특징 + 이미지 간 차별화 어휘(sub-structure vocabulary)** 분리 제공

```
[현재 KB 구조]
  "카테고리명": "공통 외관 설명 한 문장"

[개선 KB 구조]
  "카테고리명":
    "문화적 정의 + 공통 식별자"             ← 기존 유지
    "주요 하위 구조물 어휘 목록"             ← 신규 추가
    "이미지 간 차별화 단서 (계절/각도/인파)" ← 신규 추가
```

### 3-2. 건축 카테고리별 개선안 (예시)

#### 행궁(Haenggung) — 가장 심각, 우선순위 1

```python
# AS-IS
"Haenggung":
    "Haenggung, a Korean temporary royal palace; visually characterized by
     wooden buildings with curved tile roofs and dancheong-painted eaves
     arranged around courtyards."

# TO-BE (안)
"Haenggung":
    "Haenggung, a Korean temporary royal palace used during royal processions; "
    "key sub-structures: Jeongjeon (main hall), Chimjeon (royal quarters), "
    "Oenghaenggak (outer corridor), Nasammun (inner gate), Oessammun (outer gate). "
    "Common visual identifiers: dancheong-painted eaves, stone-paved courtyards, "
    "tiled roofs, stone staircases. "
    "Image-discriminating cues: which structure is most prominent (gate / hall / corridor), "
    "season (spring blossoms / summer foliage / autumn leaves / snow), "
    "presence and activity of visitors."
```

#### 경복궁(Gyeongbokgung) — 우선순위 2

```python
# TO-BE (안)
"Gyeongbokgung palace":
    "Gyeongbokgung, the main royal palace of the Joseon dynasty in Seoul; "
    "key sub-structures: Gwanghwamun (main gate), Heungnyemun (second gate), "
    "Geunjeongjeon (throne hall), Gyeonghoeru (banquet pavilion on a pond), "
    "Hyangwonjeong (hexagonal pavilion). "
    "Common visual identifiers: wooden pavilions with upturned eaves, "
    "colorful dancheong paintwork, raised stone platforms, wide stone-paved plazas. "
    "Image-discriminating cues: which gate/hall is foregrounded, time of day "
    "(daytime / night illumination), season, presence of guards in ceremonial uniform."
```

#### 팔달문(Paldalmun Gate) — 우선순위 3

```python
# TO-BE (안)
"Paldalmun Gate":
    "Paldalmun, the south gate of Hwaseong Fortress in Suwon; "
    "key features: two-story wooden pavilion atop an arched stone gateway, "
    "curved fortress walls (Ongseong) flanking both sides, "
    "surrounding modern cityscape visible in background. "
    "Image-discriminating cues: viewpoint (frontal / side / aerial), "
    "surrounding traffic/pedestrians, day/night, seasonal foliage on walls."
```

#### 남한산성(Namhansanseong) — 우선순위 4

```python
# TO-BE (안)
"Namhansanseong fortress":
    "Namhansanseong, a UNESCO mountain fortress south of Seoul; "
    "key features: long stone walls following mountain ridges, "
    "arched gates (East/West/South/North), watchtowers (Poru), "
    "inner village with traditional buildings. "
    "Image-discriminating cues: which section of wall or gate is shown, "
    "mountain ridge visibility, season (snow / autumn / spring blossoms), "
    "presence of hikers or visitors."
```

### 3-3. 프롬프트 보완 (KB 개선과 병행)

KB 어휘 제공만으로는 부족할 수 있음. 건축 카테고리에서는 아래 지시를 KB 내에 포함하거나 프롬프트에 추가:

```
"Identify which specific structure or part of {category} is most
prominent in this image, then describe it."
```

또는 KB 끝에 서술 유도문 추가:
```
"...Describe the most prominent structure visible and note
any distinctive contextual details (season, lighting, visitors)."
```

---

## 4. 실험 우선순위 및 계획

### 4-1. 핵심 원칙: 변수를 분리하여 순차 실험

KB 품질(변수 A)과 프롬프트 구조(변수 B)는 독립 변수.
둘을 동시에 바꾸면 어느 쪽 효과인지 해석 불가 → **한 번에 하나씩**.

### 4-2. 로드맵

```
[Step 1] P15 vs P19 pairwise 평가  ← 즉시 실행 가능 (캡션 이미 존재)
           ↓
           현재 KB 기반, 프롬프트 구조 차이만 비교
           → 최적 프롬프트 확정

[Step 2] 건축 카테고리 KB 개선
           ↓
           확정된 프롬프트로 건축 카테고리 캡션만 재생성
           (전체 재생성 불필요 — 건축 4개 카테고리 × 20장 = 80장)

[Step 3] KB 개선 전/후 비교 (선택)
           ↓
           개선된 캡션으로 pairwise 재평가
           또는 base vs covt 최종 비교

[Step 4] 최종 학습 데이터 구성
           ↓
           도메인별 최적 캡션 선택:
             음식·민속놀이 → 확정 프롬프트 캡션
             건축          → KB 개선 후 재생성 캡션
```

### 4-3. Step 1을 먼저 하는 이유

| 이유 | 설명 |
|------|------|
| 비용 효율 | P15, P19 캡션 이미 존재 → eval만 실행하면 됨 |
| 변수 분리 | KB 문제는 P15·P19 양쪽에 동등 적용 → 프롬프트 상대 비교 유효 |
| 방향 결정 | P19 우세 시 → P19+개선 KB / 도메인별 차이 시 → 해당 도메인 KB 집중 보강 |
| KB 재생성 범위 결정 | 어느 프롬프트가 이기는지 알아야 어떤 프롬프트로 재생성할지 확정됨 |

> KB를 먼저 고치면 P15 vs P17 비교도 무효화되고, P15 vs P19도 새로 해야 하는 연쇄 재실험 발생.

### 4-4. Step 2 실행 범위

KB 개선 후 재생성 대상: **건축 4개 카테고리 × 20장 = 80장**
음식, 민속놀이, 역사, 의례 카테고리는 현재 캡션 유지.

---

## 5. 참고: 후처리 중복 제거 (즉각 적용 가능한 보완책)

KB 개선 전이라도 현재 캡션에서 중복 제거만으로 학습 노이즈를 줄일 수 있음.

```python
# 유사도 기반 중복 제거 기준 (안)
from difflib import SequenceMatcher

def is_duplicate(cap_a, cap_b, threshold=0.85):
    return SequenceMatcher(None, cap_a, cap_b).ratio() > threshold

# 카테고리 내에서 sim > 0.85인 쌍 중 하나 제거
# → 행궁: 20장 → 약 9장으로 축소 (unique만 유지)
```

단, 이는 임시방편. 근본 해결은 KB + 프롬프트 개선으로 처음부터 다양한 캡션을 생성하는 것.

---

## 6. 요약

| 구분 | 내용 |
|------|------|
| **문제** | 건축 카테고리(특히 행궁) 내 캡션 71.2% unique → 중복 캡션이 파인튜닝 시 visual grounding 붕괴 유발 |
| **원인** | KB가 카테고리 공통 정적 특징만 기술 → 이미지 간 차별화 어휘 부재 |
| **개선 방향** | KB에 sub-structure 어휘 + 이미지 차별화 단서 추가 + 프롬프트에 "어느 구조물이 보이는지" 식별 지시 추가 |
| **실험 순서** | ① P15 vs P19 pairwise (즉시) → ② 프롬프트 확정 → ③ 건축 KB 개선 + 80장 재생성 → ④ 최종 비교 |
| **주의** | KB와 프롬프트를 동시에 바꾸면 효과 해석 불가 → 반드시 순차 실험 |
