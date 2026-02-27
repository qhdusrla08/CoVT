# Knowledge Base(KB) 주입 실험 – 작업 기록

## 1. 실험 목적

기존 캡션 생성 파이프라인(P0–P6)은 이미지 디렉토리명(`{category}`)만을 프롬프트 힌트로 제공한다.
디렉토리명은 고유명사(예: `Kimjang`, `Yutnori`)이므로, VLM이 해당 개념의 **문화적 맥락을 모를 경우** 부정확하거나 피상적인 캡션을 생성할 수 있다.

이 실험은 카테고리별로 짧은 **문화적 정의(Definition, 1문장)** 를 추가 힌트로 주입했을 때, 아래 항목의 변화를 측정하는 것을 목표로 한다.

| 평가 축 | 기대 효과 |
|---|---|
| 문화/도메인 오류 감소 | "일본 전통" 등 잘못된 문화 귀속 방지 |
| 캡션 품질 향상 | 시각 묘사 + 문화적 맥락의 정확한 통합 |
| KB Miss 시 안전한 Fallback | 정의가 없는 카테고리는 기존 동작(카테고리명만 사용) 유지 |

---

## 2. 작업 내역

### 2-1. KNOWLEDGE_BASE 딕셔너리 구축

`caption_gen_korean.py` 상단에 `KNOWLEDGE_BASE: Dict[str, str]`을 추가하였다.

- **키(Key)**: `extract_category_from_path()` 함수의 반환값과 일치하는 카테고리명
- **값(Value)**: 해당 문화 개념에 대한 영어 1문장 정의

총 **32개 카테고리 전수 커버** (B_korean 트랙 전체).

#### KB 전체 목록

| # | 분류 | 카테고리 키 | 정의 (요약) |
|---|---|---|---|
| 9001 | 전통 문화·행사 | Kimjang | 늦가을 김치를 대량으로 담그는 한국 공동체 전통 |
| 9002 | 전통 문화·행사 | Independence movement | 1919년 3·1운동으로 상징되는 항일 독립운동 |
| 9003 | 전통 문화·행사 | Traditional wedding | 한복·목안·유교 의례를 갖춘 전통혼례 |
| 9004 | 전통 문화·행사 | Ancestral rite | 제사(祭祀), 유교식 조상 추모 의례 |
| 9005 | 전통 문화·행사 | Seongmyo | 추석·한식에 조상 묘를 찾아 돌보는 성묘 의례 |
| 9011 | 전통 문화·행사 | Jeol and Sebae | 절과 세배 – 연하자가 연장자에게 올리는 큰절 |
| 9013 | 전통 문화·행사 | Kite flying | 연날리기(Yeon-nalligi), 설날 전후의 전통 놀이 |
| 9014 | 전통 문화·행사 | Yutnori | 윷놀이, 설날에 즐기는 전통 보드게임 |
| 9015 | 전통 문화·행사 | Ganggangsullae | 보름달 아래 여성들이 추는 강강술래 (UNESCO) |
| 9016 | 전통 문화·행사 | Mask Dance | 탈춤(Talchum), 풍자와 유머의 가면극 |
| 9032 | 전통 문화·행사 | Jegichagi | 제기차기, 종이 제기를 차올리는 전통 놀이 |
| 9010 | 역사 인물 | Admiral Yi Sun-shin | 조선 수군 명장, 거북선 승리로 유명 |
| 9012 | 역사 인물 | Sejongdaewang | 세종대왕, 한글 창제 |
| 9031 | 역사 인물 | The Korean War | 1950–1953 한국전쟁, 휴전으로 분단 고착 |
| 9006 | 역사 건축·유적 | Independence hall of Korea | 천안 독립기념관 |
| 9007 | 역사 건축·유적 | Cheomseongdae observatory | 경주 첨성대, 동아시아 최고(最古) 천문대 |
| 9008 | 역사 건축·유적 | Gyeongbokgung palace | 경복궁, 조선 정궁 (1395) |
| 9009 | 역사 건축·유적 | Cheonggyecheon | 청계천, 2005년 복원된 도심 하천 |
| 9027 | 역사 건축·유적 | Namhansanseong fortress | 남한산성, UNESCO 세계유산 산성 |
| 9028 | 역사 건축·유적 | Haenggung | 행궁, 수원 화성행궁 (정조) |
| 9029 | 역사 건축·유적 | Paldalmun Gate | 팔달문, 수원 화성 남문 (UNESCO) |
| 9030 | 역사 건축·유적 | EXPO Hanbit Tower | 1993 대전엑스포 한빛탑 |
| 9017 | 한국 음식 | Ramyeon | 한국식 매운 라면 (일본 라멘과 구분) |
| 9018 | 한국 음식 | Gimbap | 김밥, 김·밥·야채·계란·고기 말이 |
| 9019 | 한국 음식 | Gyeranmari | 계란말이, 야채를 넣은 한국식 롤 오믈렛 |
| 9020 | 한국 음식 | Panfried battered meatballs | 완자전, 두부·고기 경단에 계란옷을 입혀 부침 |
| 9021 | 한국 음식 | Hangwa | 한과, 곡물·꿀·과일로 만든 전통 과자 |
| 9022 | 한국 음식 | Yaksik | 약식, 찹쌀·밤·대추·잣·참기름 떡 |
| 9023 | 한국 음식 | Yakgwa | 약과, 밀가루·참기름·꿀로 만든 유밀과 |
| 9024 | 한국 음식 | Bulgogi | 불고기, 간장·설탕·참기름·마늘 양념 구이 |
| 9025 | 한국 음식 | Tteokguk and Mandu soup | 떡국·만두국, 설날 맑은 장국 |
| 9026 | 한국 음식 | Japchae | 잡채, 당면·야채·고기 볶음 |

### 2-2. P7 프롬프트 추가

기존 프롬프트 체계(P0–P6)에 **P7**을 추가하였다.

```
P6 (기존): "Describe the visual appearance and cultural context of '{category}' in **exactly one sentence**."
P7 (신규): "Describe the visual appearance and cultural context of '{category}' ({definition}) in **exactly one sentence**."
```

| 비교 항목 | P6 | P7 |
|---|---|---|
| 카테고리명 힌트 | O | O |
| 문화적 정의 힌트 | X | O (KB에서 조회) |
| 실험 변수 | 대조군 | 처리군 |

### 2-3. 프롬프트 포매팅 로직 수정

`main()` 함수의 프롬프트 주입 부분을 확장하여 `{definition}` 플레이스홀더를 처리한다.

```python
# 변경 전
if "{category}" in prompt_template:
    current_prompt = prompt_template.format(category=category_name)
else:
    current_prompt = prompt_template

# 변경 후
if "{definition}" in prompt_template:
    definition = KNOWLEDGE_BASE.get(category_name, category_name)  # fallback: 카테고리명
    current_prompt = prompt_template.format(
        category=category_name, definition=definition
    )
elif "{category}" in prompt_template:
    current_prompt = prompt_template.format(category=category_name)
else:
    current_prompt = prompt_template
```

- `KNOWLEDGE_BASE`에 키가 존재하면 → 문화적 정의를 주입
- 키가 없으면(KB miss) → 카테고리명 자체를 fallback으로 사용 (기존 P6과 동일 효과)

### 2-4. 키 표기 정규화 및 전체 파일 동기화

표준 로마자 표기법에 맞게 두 카테고리의 키를 수정하고, 레포 전체에 일괄 반영하였다.

| 변경 전 | 변경 후 | 사유 |
|---|---|---|
| `Yunnori` | `Yutnori` | 표준 로마자 표기법 |
| `New Year's bow and Deep bow` | `Jeol and Sebae` | 절(Jeol)과 세배(Sebae)의 포괄적 의미 반영 |

**변경 범위:**

| 대상 | 변경 사항 |
|---|---|
| 데이터 디렉토리 (2건) | `9014_Yunnori` → `9014_Yutnori`, `9011_New Year's bow_Deep bow` → `9011_Jeol_Sebae` |
| `caption_gen_korean.py` | KNOWLEDGE_BASE 키 수정 |
| `outputs_korean/*.jsonl` (14파일) | `category`, `image_path`, `caption` 내 키값 치환 |
| `eval_outputs_korean/*.jsonl` (3파일) | 동일 치환 |
| `B_korean_eval_analysis.md` (1파일) | 텍스트 내 키값 치환 |

치환 후 `Yunnori` / `New Year's bow` 패턴의 잔여 항목: **0건** (전수 확인 완료).

---

## 3. 동작 검증 결과

32개 전체 카테고리에 대해 KB 키 매칭 테스트를 수행하였다.

```
=== KB 키 매칭 테스트 (32개 디렉토리) ===
  [OK] 9001_Kimjang            -> "Kimjang"
  [OK] 9011_Jeol_Sebae         -> "Jeol and Sebae"
  [OK] 9014_Yutnori            -> "Yutnori"
  [OK] 9025_Tteokguk_Mandu soup -> "Tteokguk and Mandu soup"
  ... (32개 전체 OK)

  MISS: 0건
```

P7 프롬프트 포매팅 예시:

```
[입력] category="Tteokguk and Mandu soup"
[P7]  Describe the visual appearance and cultural context of
      'Tteokguk and Mandu soup' (A Korean New Year soup combining
      sliced rice cakes (tteok) and dumplings (mandu) in a clear broth.)
      in **exactly one sentence**.
```

---

## 4. 실험 실행 방법

```bash
# P7 (KB 주입) 캡션 생성
python gradio/caption_gen_korean.py \
  --track_dirs B=gradio/data/B_korean \
  --prompt_id P7 \
  --model_tag covt \
  --ckpt <checkpoint_path> \
  --recursive \
  --skip_existing

# P6 (대조군) 캡션 생성
python gradio/caption_gen_korean.py \
  --track_dirs B=gradio/data/B_korean \
  --prompt_id P6 \
  --model_tag covt \
  --ckpt <checkpoint_path> \
  --recursive \
  --skip_existing
```

## 5. 기대 비교 분석

| 비교 쌍 | 목적 |
|---|---|
| P6 vs P7 (동일 모델) | KB 정의 주입의 순수 효과 측정 |
| P7 base vs P7 covt | KB 주입이 base/CoVT 모델에 미치는 차등 효과 |
| P0 vs P7 | 카테고리 힌트 없음 → KB 정의까지 제공 시 최대 변화폭 |

---

## 6. 수정 파일 목록

| 파일 | 변경 유형 |
|---|---|
| `gradio/caption_gen_korean.py` | KB 딕셔너리 추가, P7 프롬프트 추가, 포매팅 로직 확장, `--prompt_id` choices 확장 |
| `gradio/data/B_korean/9014_Yutnori/` | 디렉토리명 변경 (← `9014_Yunnori`) |
| `gradio/data/B_korean/9011_Jeol_Sebae/` | 디렉토리명 변경 (← `9011_New Year's bow_Deep bow`) |
| `gradio/outputs_korean/*.jsonl` (14파일) | 키 표기 정규화 치환 |
| `gradio/eval_outputs_korean/*.jsonl` (3파일) | 키 표기 정규화 치환 |
| `gradio/eval_outputs_korean/p6_covt_win_summary.txt` | 키 표기 정규화 치환 |
| `B_korean_eval_analysis.md` | 키 표기 정규화 치환 |
