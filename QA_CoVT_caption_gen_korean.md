# CoVT & caption_gen_korean.py Q&A 정리

## Q1. caption_gen_korean.py가 CoVT를 제대로 적용하고 있는가?

**A: 제대로 적용하고 있다.**

- CoVT는 프롬프트 기법이 아니라, **학습(training) 단계에서 모델 가중치를 변경하는 방법론**이다.
- 학습 시: SAM, DINO, DepthAnything 등의 앵커 모델에서 추출한 시각적 특징을 특수 토큰(`<|sam_pad|>`, `<|depth_pad|>` 등)으로 주입하고, projection layer + cross-attention으로 LLM 임베딩과 정렬시킴.
- 추론(inference) 시: 특수 토큰이나 특별한 디코딩이 **필요 없음**. 학습된 시각적 추론 능력이 이미 **모델 가중치에 내재화**되어 있어서, 일반 VLM처럼 `model.generate()`를 호출하면 됨.
- `caption_gen_korean.py`의 추론 방식은 레포의 공식 Gradio 데모(`gradio_demo.py`)와 동일하다.
- 프롬프트(P0~P4)는 CoVT 자체의 구현이 아니라 **프롬프트 수준의 실험 변수**이다. CoVT의 핵심은 `--ckpt` 또는 `--model_name`으로 로드하는 학습된 모델 가중치에 있다.

---

## Q2. 추론을 위해 SAM, DepthAnything V2, DINO의 가중치가 필요한가?

**A: 필요 없다.**

- CoVT 모델 코드(`covt_qwen2_5_vl.py`)에서 `self.anchor_models = None`이 초기값이며, 학습 시에만 `get_anchor_model_ids()`를 호출하여 앵커 모델을 로드한다.
- 추론 시에는 `anchor_models`가 `None` 상태로 유지되어 앵커 모델 관련 로직이 전부 스킵된다.
- 앵커 모델들은 학습 과정에서 **teacher 역할**만 수행하며, 학습이 끝나면 버려진다.

| 단계 | anchor_models | SAM/DINO/Depth 필요 여부 |
|------|--------------|------------------------|
| 학습 시 | `get_anchor_model_ids()` 호출 → 앵커 모델 로드 | **필요** (GT 특징 추출용) |
| 추론 시 | `None` 상태 유지 | **불필요** |

---

## Q3. HuggingFace에서 체크포인트를 제대로 불러오고 있는지 확인하려면?

**A: `load_model_and_processor` 함수 (line 64-100)를 확인한다.**

```python
# line 71: 실제 로드 소스 결정
src = ckpt if ckpt is not None else model_name

# line 74-79: 모델 로드
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(src, ...)

# line 81-86: 프로세서 로드
processor = AutoProcessor.from_pretrained(src, ...)
```

- HuggingFace `config.json`의 `"architectures": ["Qwen2_5_VLForConditionalGeneration"]`과 코드의 클래스가 **정확히 일치**.
- `auto_map` 필드가 없으므로 커스텀 클래스 매핑 이슈도 없음.
- 체크포인트는 CoVT 학습 후 **표준 Qwen2.5-VL 구조로 저장**된 것이며, 정상적으로 로드됨.

---

## Q4. --ckpt 없이 --model_name에 CoVT 체크포인트를 넣으면 CoVT가 적용되는가?

**A: 적용된다.**

```bash
python caption_gen_korean.py \
  --track_dirs B=data/B_korean \
  --model_name Wakals/CoVT-7B-seg_depth_dino \
  --model_tag covt \
  --prompt_id P0 \
  ...
```

- `--ckpt`를 안 줬으므로 `src = model_name = "Wakals/CoVT-7B-seg_depth_dino"`가 됨.
- 이 HuggingFace 모델이 곧 CoVT 학습이 완료된 체크포인트이므로 CoVT 가중치가 그대로 로드됨.

---

## Q5. 모델 로드 시 "unused weights" 경고가 뜨는데 무시해도 되는가?

**A: 무시해도 된다.**

경고에서 drop되는 가중치 목록:
- `sam_projection`, `sam_cross_attention`, `sam_query_vectors`
- `dino_projection`, `dino_cross_attention`, `dino_query_vectors`
- `depth_projection`, `depth_cross_attention`, `depth_query_vectors`, `depth_token_generator`

이유:
1. 체크포인트에는 CoVT 학습 전용 레이어가 포함되어 있지만, `Qwen2_5_VLForConditionalGeneration`에는 해당 레이어가 정의되어 있지 않아 무시됨.
2. CoVT 학습 3단계 구조에서 Stage 3(vqa_only_stage 이후)에서는 visual thinking token 없이 **LLM 본체만** fine-tuning함.
3. 추론 시 입력에 visual thinking token이 없으므로 projection/cross-attention이 실행될 경로 자체가 없음.
4. **LLM 백본 가중치**(LoRA가 merge된 상태)는 정상 로드되므로 추론에 문제없음.
