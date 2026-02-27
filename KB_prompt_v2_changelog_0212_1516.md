# KB & Prompt v2 변경 사항 정리

## 1. 변경 배경

P7~P13 GPT 평가(eval_results_P7-13.jsonl, 350건) 분석 결과 도출된 문제점:

| 문제 | 원인 | 영향 |
|---|---|---|
| CoVT 할루시네이션 | KB 정의의 문화/역사 서술을 이미지에 없는데도 그대로 출력 | P11(26%), P13(40%) 등 base 대비 열세 |
| 메타 용어 출력 | 프롬프트의 "segmentation, depth, patch features" 지시를 캡션에 그대로 삽입 | "showcasing the depth and patch features of..." 등 부자연스러운 캡션 |
| verify형 지시의 부작용 | "verify visual features" 지시가 보이지 않는 세부사항까지 확인하려는 경향 유발 | P11 할루시네이션 최다 (CoVT 13회 패널티) |

---

## 2. KB 수정: 시각적 공통 특징 중심으로 정제

### 2.1 수정 원칙

1. **"visually characterized by..." 형식 통일** — 모든 항목에 시각적 특징 서술 도입
2. **과잉 구체 묘사 제거** — 이미지마다 달라질 수 있는 특정 색상/소재/배치/토핑 등 축소
3. **비시각 정보 최소화** — 연도, 유래, 의미 등 역사/문화 배경 서술 제거
4. **공통 식별 특징만 유지** — 해당 카테고리의 거의 모든 이미지에서 관찰 가능한 특징만 잔존

### 2.2 항목별 변경 내역

#### 전통 문화/행사

| 카테고리 | 기존 (v1) | 변경 (v2) | 변경 사유 |
|---|---|---|---|
| **Kimjang** | "A traditional Korean communal practice of making and sharing large quantities of kimchi in late autumn to prepare for winter." | "Kimjang, a Korean communal kimchi-making practice; visually characterized by groups of people in aprons and gloves handling salted napa cabbage and red chili paste (gochugaru) at large tables." | 문화 설명 → 시각 특징(앞치마, 장갑, 배추, 고춧가루)으로 전환 |
| **Independence movement** | "The Korean independence movement against Japanese colonial rule, symbolized by the March 1st Movement of 1919." | "The Korean independence movement; visually characterized by people in white hanbok or period clothing carrying or waving the Taegeukgi (Korean flag)." | 역사 서술(3.1운동, 일제) 제거, 시각 요소(백색 한복, 태극기)만 유지 |
| **Traditional wedding** | "A Korean traditional wedding ceremony (Jeontonghonrye) featuring colorful hanbok attire, a wooden goose, and Confucian rituals." | "Jeontonghonrye, a Korean traditional wedding; visually characterized by a couple in colorful ceremonial hanbok set before a ritual table, often with wooden mandarin ducks and candles." | "Confucian rituals" 등 비시각 정보 제거, 특정 색상(green wonsam, blue dallyeong) 일반화 |
| **Ancestral rite** | "Jesa, a Korean Confucian memorial ceremony honoring deceased ancestors with a ritual table of food offerings." | "Jesa, a Korean ancestral memorial rite; visually characterized by a ritual table with neatly arranged food offerings and family members performing deep bows." | 개별 제물(rice, soup, fruit, dried fish) 나열 제거 |
| **Seongmyo** | "A Korean Confucian rite of visiting and tending ancestral graves, typically performed during Chuseok or Hansik." | "Seongmyo, a Korean grave-tending rite; visually characterized by families gathered around a grass-covered burial mound with food offerings placed before the grave." | 시기 정보(추석, 한식) 제거, 행위 다양성(bowing, trimming grass) 축소 |
| **Jeol and Sebae** | "Korean formal deep bows (jeol) performed by juniors to elders or the deceased, including Sebae..." | "Jeol and Sebae, Korean formal deep bows; visually characterized by a person kneeling and bowing deeply with hands on the floor, often wearing hanbok." | 관계 설명(juniors to elders) 제거, 장소(wooden floors, traditional interior) 축소 |
| **Kite flying** | "Yeon-nalligi, a traditional Korean pastime of flying rectangular kites, especially during the Lunar New Year period." | "Yeon-nalligi, Korean traditional kite flying; visually characterized by rectangular or shield-shaped kites with a central hole and colorful geometric patterns." | 시기(Lunar New Year) 제거, 색상 나열(red, blue, yellow) 축소 |
| **Yutnori** | "Yutnori, a traditional Korean board game played with four wooden sticks, popular during Lunar New Year gatherings." | "Yutnori, a Korean traditional board game; visually characterized by four wooden sticks (yut) and a game board with circular stations, played by groups seated on the floor." | 시기 제거, "short cylindrical" 등 과잉 형태 묘사 축소 |
| **Ganggangsullae** | "A Korean circle dance and folk song performed by women under the full moon, designated as UNESCO Intangible Cultural Heritage." | "Ganggangsullae, a Korean circle dance; visually characterized by women in hanbok holding hands in a large circle formation outdoors." | UNESCO 지정, 보름달 등 비시각 정보 제거 |
| **Mask Dance** | "Talchum, a traditional Korean masked dance-drama that satirizes social classes through choreography and humorous dialogue." | "Talchum, a Korean masked dance-drama; visually characterized by performers wearing exaggerated painted masks and loose colorful robes." | 사회 풍자 설명 제거, 재질(wooden/papier-mache) 축소 |
| **Jegichagi** | "A traditional Korean shuttlecock-kicking game similar to hacky sack, played by keeping a paper-weighted jegi in the air." | "Jegichagi, a Korean shuttlecock-kicking game; visually characterized by a small paper-and-coin shuttlecock (jegi) kicked into the air with the foot." | "similar to hacky sack" 비교 제거, 발 부위(inner side) 축소 |

#### 역사 인물

| 카테고리 | 기존 | 변경 | 변경 사유 |
|---|---|---|---|
| **Admiral Yi Sun-shin** | "A revered Korean naval commander of the Joseon Dynasty, famous for his turtle ship (Geobukseon) victories against Japan." | "Admiral Yi Sun-shin, a Korean naval hero; visually characterized by statues in Joseon military armor with a sword, or turtle ship (Geobukseon) models with a dragon-head prow." | 역사 서술 → 동상/거북선 시각 특징 |
| **Sejongdaewang** | "King Sejong the Great of the Joseon Dynasty, celebrated for creating Hangul, the Korean phonetic alphabet." | "King Sejong the Great; visually characterized by a seated figure in royal Joseon robes with a winged crown (ikseongwan), often with Hangul script nearby." | 업적 서술 → 좌상/복식 시각 특징 |
| **The Korean War** | "The 1950-1953 conflict between North and South Korea, which ended in an armistice and left the peninsula divided." | "The Korean War (1950-1953); visually characterized by soldier statues or monuments, memorial walls with engraved names, or wartime photographs." | 정치 배경 제거 → 기념물/사진 시각 특징 |

#### 역사 건축/유적

| 카테고리 | 기존 | 변경 | 변경 사유 |
|---|---|---|---|
| **Independence hall** | "A memorial and museum in Cheonan commemorating Korea's resistance..." | "visually characterized by a large traditional-style building with a grand curved roof, open plazas, and Korean flags." | 위치(천안), 역사 배경 제거 |
| **Cheomseongdae** | "A 7th-century stone astronomical observatory in Gyeongju, one of the oldest..." | "visually characterized by a cylindrical stacked-stone tower with a square top frame, standing in an open field." | 연대, 지명, 비교(oldest in East Asia) 제거, 크기(9m), 방향(south-facing) 축소 |
| **Gyeongbokgung** | "The main royal palace of the Joseon Dynasty in Seoul, originally built in 1395..." | "visually characterized by wooden pavilions with upturned eaves, colorful dancheong paintwork, and raised stone platforms." | 연대, 전각명(Geunjeongjeon), 색상 나열(red,green,blue,gold) 축소 |
| **Cheonggyecheon** | "A restored urban stream in central Seoul, transformed from a covered highway..." | "visually characterized by a shallow waterway between stone embankments with pedestrian walkways alongside." | 역사(2005년 복원, 고가도로) 제거, stepping stones 등 축소 |
| **Namhansanseong** | "A UNESCO World Heritage mountain fortress south of Seoul..." | "visually characterized by long stone walls along mountain ridges with arched gates and watchtowers." | UNESCO, 백제/조선 역사 제거, 숲 묘사 축소 |
| **Haenggung** | "A temporary royal palace outside the capital, most notably Hwaseong Haenggung..." | "visually characterized by wooden buildings with curved tile roofs and dancheong-painted eaves arranged around courtyards." | 정조, 수원 등 역사 제거, "fortress complex" 축소 |
| **Paldalmun** | "The southern gate of Hwaseong Fortress in Suwon, part of the UNESCO..." | "visually characterized by an arched stone gateway topped with a wooden pavilion, flanked by curved fortress walls." | UNESCO, "southern", 로터리 등 축소 |
| **EXPO Hanbit Tower** | "A landmark observation tower built for the 1993 Daejeon Expo..." | "visually characterized by a tall white observation tower with a disc-shaped deck and pointed spire." | 연대(1993), 과학기술 상징 등 제거 |

#### 한국 음식

| 카테고리 | 주요 제거 내용 | 사유 |
|---|---|---|
| **Ramyeon** | "whole egg, sliced scallions, aluminum/steel pot (naembi)" | 토핑/용기가 이미지마다 다름 |
| **Gimbap** | "yellow pickled radish, orange carrot, green spinach, egg, meat" | 속재료 구성이 다양함 |
| **Gyeranmari** | "green scallion or carrot pieces" | 속재료 다양함 |
| **Panfried battered meatballs** | "arranged on a plate, smooth pan-seared surface" | 플레이팅 다양함 |
| **Hangwa** | "often arranged neatly on a platter or in ornate boxes" | 용기 다양함 |
| **Yaksik** | "whole chestnuts, red jujubes", "caramelized sheen from sesame oil and brown sugar" | "whole/red" 등 세부 묘사 축소 |
| **Yakgwa** | "often garnished with pine nuts on top" | 토핑 없는 경우도 있음 |
| **Bulgogi** | "sizzling on a grill or domed pan, accompanied by lettuce leaves and banchan" | 조리 상태/곁들임 다양함 |
| **Tteokguk/Mandu** | "garnished with shredded egg strips and sliced scallions" | 고명 다양함 |
| **Japchae** | "(orange carrot, green spinach, yellow onion, black mushroom strips)" | 개별 채소 나열 과잉 |

---

## 3. 프롬프트 v2: KB 주입 프롬프트 전면 교체

### 3.1 기존 프롬프트의 문제점 (P7~P14 v1)

| 기존 프롬프트 | 문제 |
|---|---|
| P9, P10, P11, P13 — "segmentation, depth, and patch features" | 모델이 메타 용어를 캡션에 그대로 출력 |
| P11 — "verify visual features" | 보이지 않는 특징까지 확인하려는 할루시네이션 유발 |
| P7 — "visual appearance and cultural context" | cultural context 지시가 비시각 정보 출력 유도 |
| P8 — "explain how it relates to" | 설명(explain) 지시가 캡션이 아닌 해설문 생성 유도 |

### 3.2 신규 프롬프트 설계 원칙

1. **seg/depth/patch 언급 완전 배제**
2. **"verify", "explain" 등 검증/해설형 동사 배제**
3. **KB는 "참조/가이드"로 역할 제한, 복사 방지**
4. **Grounding 제약 강도를 변수로 설계** (약 → 강)
5. **단일 문장 생성을 유도하는 구조** (두 단계 지시 금지)

### 3.3 신규 프롬프트 (P7~P12 v2)

| Prompt | 전략 | KB 역할 | Grounding 강도 |
|---|---|---|---|
| **P7** | KB 참조 + 시각 묘사 (가장 단순) | 괄호 내 보조 정보 | 약 (암묵적) |
| **P8** | 명시적 grounding 제약 | "Reference"로 분리 제공 | **강** |
| **P9** | KB를 naming guide로 제공 | 용어 사전 역할 명시 | 중 |
| **P10** | 조건부 매칭 — 보이는 것만 KB 연결 | 정의 제공 후 필터링 | **강** |
| **P11** | 이미지 우선, KB 보조 | 후순위 참조 | **강** |
| **P12** | 간결한 통합 — P7 + grounding 한 줄 | 괄호 내 보조 정보 | 중 |

#### 프롬프트 전문

```
P7:  "Describe the visual appearance of '{category}' ({definition}) in this image
      in **exactly one sentence**."

P8:  "Reference: {category} — {definition}.
      Describe only what you actually see in this image in **exactly one sentence**.
      Do not include details from the reference that are not visible."

P9:  "Use the following as a naming guide: {category} — {definition}.
      List the observable visual elements in this image in **exactly one sentence**,
      using the correct Korean cultural terms where applicable."

P10: "Given that {category} is {definition},
      describe only the features from this definition that are actually visible
      in the image in **exactly one sentence**."

P11: "Describe what you see in this image, identifying elements that relate to
      '{category}' ({definition}) in **exactly one sentence**.
      Only mention cultural terms if visually supported."

P12: "Describe the visual appearance of '{category}' ({definition}) shown in this image
      in **exactly one sentence**. Only describe what is directly observable."
```

### 3.4 예상 실험 결과 가설

- **P7** (약한 grounding): v1 P7(66%)과 유사하거나 소폭 향상 예상. KB가 시각적으로 정제되어 할루시네이션 감소 기대.
- **P8, P10** (강한 grounding): 할루시네이션 억제 효과 극대화. 다만 KB 활용도가 낮아져 base와 차별화 약화 가능성.
- **P9** (naming guide): KB를 용어 사전으로만 활용하므로 정확한 문화 용어 사용 + grounding 균형 기대.
- **P11** (이미지 우선): v1 P11(26%)에서 대폭 개선 예상. "verify" 제거 + 이미지 우선 구조.
- **P12** (간결 통합): P7과 유사하되 grounding 한 줄 추가로 소폭 안전장치.

---

## 4. 다음 단계

1. 수정된 KB + 프롬프트 v2로 캡션 재생성 (P7~P12, base/covt 각각)
2. GPT 평가 재수행 (gpt-4.1 업그레이드 검토)
3. v1 vs v2 비교 분석
4. Best prompt 선정 → 대규모 캡션 생성 → fine-tuning 데이터 확정
