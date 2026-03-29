import os
import json
import time
import argparse
from typing import Dict, Any, List, Optional, Tuple
import re
import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from typing import Optional

# ---------------- Knowledge Base (KB) ----------------
# 카테고리명 → 짧은 영어 문화적 정의 (1문장)
# extract_category_from_path()의 반환값을 키로 사용

KNOWLEDGE_BASE: Dict[str, str] = {
    # ── 전통 문화·행사 ──
    "Kimjang":
        "Kimjang, a Korean communal kimchi-making practice; visually characterized by groups of people in aprons and gloves "
        "handling salted napa cabbage and red chili paste (gochugaru) at large tables.",
    "Independence movement":
        "The Korean independence movement; visually characterized by people in white hanbok or period clothing "
        "carrying or waving the Taegeukgi (Korean flag).",
    "Korean traditional wedding":
        "Jeontonghonrye, a Korean traditional wedding; visually characterized by a couple in colorful ceremonial hanbok "
        "set before a ritual table, often with wooden mandarin ducks and candles.",
    "Ancestral rite":
        "Jesa, a Korean ancestral memorial rite; visually characterized by a ritual table with neatly arranged food offerings "
        "and family members performing deep bows.",
    "Seongmyo":
        "Seongmyo, a Korean grave-tending rite; visually characterized by families gathered around a grass-covered burial mound "
        "with food offerings placed before the grave.",
    "Jeol and Sebae":
        "Jeol and Sebae, Korean formal deep bows; visually characterized by a person kneeling and bowing deeply with hands on the floor, "
        "often wearing hanbok.",
    "Kite flying":
        "Yeon-nalligi, Korean traditional kite flying; visually characterized by rectangular or shield-shaped kites with a central hole "
        "and colorful geometric patterns.",
    "Yutnori":
        "Yutnori, a Korean traditional board game; visually characterized by four wooden sticks (yut) and a game board "
        "with circular stations, played by groups seated on the floor.",
    "Ganggangsullae":
        "Ganggangsullae, a Korean circle dance; visually characterized by women in hanbok holding hands in a large circle formation outdoors.",
    "Mask Dance":
        "Talchum, a Korean masked dance-drama; visually characterized by performers wearing exaggerated painted masks "
        "and loose colorful robes.",
    "Jegichagi":
        "Jegichagi, a Korean shuttlecock-kicking game; visually characterized by a small paper-and-coin shuttlecock (jegi) "
        "kicked into the air with the foot.",

    # ── 역사 인물 ──
    "Admiral Yi Sun-shin":
        "Admiral Yi Sun-shin, a Korean naval hero; visually characterized by statues in Joseon military armor with a sword, "
        "or turtle ship (Geobukseon) models with a dragon-head prow.",
    "Sejongdaewang":
        "King Sejong the Great; visually characterized by a seated figure in royal Joseon robes with a winged crown (ikseongwan), "
        "often with Hangul script nearby.",
    "The Korean War":
        "The Korean War (1950-1953); visually characterized by soldier statues or monuments, memorial walls with engraved names, "
        "or wartime photographs.",

    # ── 역사 건축·유적 ──
    "Independence hall of Korea":
        "Independence Hall of Korea; visually characterized by a large traditional-style building with a grand curved roof, "
        "open plazas, and Korean flags.",
    "Cheomseongdae observatory":
        "Cheomseongdae, a Korean stone observatory; visually characterized by a cylindrical stacked-stone tower "
        "with a square top frame, standing in an open field.",
    "Gyeongbokgung palace":
        "Gyeongbokgung, a Korean royal palace; visually characterized by wooden pavilions with upturned eaves, "
        "colorful dancheong paintwork, and raised stone platforms — "
        "structures include a grand throne hall, a pavilion on a pond, and various gateways.",
    "Cheonggyecheon":
        "Cheonggyecheon, a restored urban stream in Seoul; visually characterized by a shallow waterway between stone embankments "
        "with pedestrian walkways alongside.",
    "Namhansanseong fortress":
        "Namhansanseong, a Korean mountain fortress; visually characterized by long stone walls along mountain ridges "
        "with arched gates and watchtowers. Inner-site elements include a stone pavilion or temple building.",
    "Haenggung":
        "Haenggung, a Korean temporary royal palace; visually characterized by wooden buildings with curved tile roofs "
        "and dancheong-painted eaves around stone courtyards — "
        "structures include a main hall, inner gates, and connecting corridors.",
    "Paldalmun Gate":
        "Paldalmun, a gate of Hwaseong Fortress in Suwon; visually characterized by an arched stone gateway topped with "
        "a two-story wooden pavilion, flanked by curved stone walls.",
    "EXPO Hanbit Tower":
        "Hanbit Tower, a Korean landmark; visually characterized by a tall white observation tower with a disc-shaped deck and pointed spire.",

    # ── 한국 음식 ──
    "Ramyeon":
        "Ramyeon, Korean instant noodles; visually characterized by yellow curly noodles in an orange-red spicy broth, "
        "often served in a small pot.",
    "Gimbap":
        "Gimbap, Korean seaweed rice rolls; visually characterized by cylindrical rolls of rice and colorful fillings "
        "wrapped in dark-green seaweed, sliced into round pieces.",
    "Gyeranmari":
        "Gyeranmari, Korean rolled omelette; visually characterized by a rectangular roll of cooked egg with visible layers, "
        "sliced crosswise to reveal a spiral pattern.",
    "Panfried battered meatballs":
        "Wanja-jeon, Korean pan-fried meatballs; visually characterized by small round golden-brown patties with an egg-batter coating.",
    "Hangwa":
        "Hangwa, traditional Korean confections; visually characterized by small decorative pieces in pastel colors "
        "with pressed floral or geometric patterns.",
    "Yaksik":
        "Yaksik, a Korean sweet rice dish; visually characterized by a dark-brown glossy mound of sticky glutinous rice "
        "with chestnuts, jujubes, and pine nuts.",
    "Yakgwa":
        "Yakgwa, a Korean honey cookie; visually characterized by small flower-shaped or round deep-fried pastries "
        "with a dark golden-brown glossy glaze.",
    "Bulgogi":
        "Bulgogi, Korean marinated grilled beef; visually characterized by thin slices of meat with a glossy dark-brown glaze, "
        "often on a grill or plate.",
    "Tteokguk and Mandu soup":
        "Tteokguk and Mandu soup, a Korean New Year dish; visually characterized by a clear broth with white oval sliced rice cakes (tteok) "
        "and crescent-shaped dumplings (mandu).",
    "Japchae":
        "Japchae, Korean stir-fried glass noodles; visually characterized by translucent brown noodles mixed with colorful julienned vegetables, "
        "with a glossy sesame-oil sheen.",
}

# ---------------- Prompts (MVP) ----------------


PROMPTS = {
    # ── 원본 P0~P6 (B_korean_eval_analysis.md §1.4) ──
    # P0: 대조군 — 카테고리 정보 없음
    "P0": "Describe the scene and main objects in **exactly one sentence**.",
    # P1: 기본 지시 — 카테고리명 제공 + 외양 묘사
    "P1": "Visually describe '{category}' and its layout in **exactly one sentence**.",
    # P2: 특징 추출 — 시각 요소 식별
    "P2": "Identify the visual elements of '{category}' in **exactly one clear sentence**.",
    # P3: 구조 검증 — 형태/배치 검증
    "P3": "Describe '{category}' by verifying its shapes and features in **exactly one sentence**.",
    # P4: 기술 지표 — SAM/Depth/DINO 지표 활용 지시
    "P4": "Describe '{category}' using segmentation, depth, and patch features in **exactly one sentence**.",
    # P5: 지표 통합 — 시각 지표 직접 반영
    "P5": "Visually describe '{category}' incorporating segmentation, depth, and patch features in **exactly one sentence**.",
    # P6: 시각-지식 통합 — 외양 묘사 + 문화적 맥락 결합
    "P6": "Describe the visual appearance and cultural context of '{category}' in **exactly one sentence**.",

    # ── KB 주입 프롬프트 v2 (시각적 KB + grounding 제약) ──

    # P7: KB 참조 + 시각 묘사 (가장 단순한 형태)
    # 전략: KB 정의를 괄호로 제공하되, 시각 묘사에 집중하도록 유도
    "P7": "Describe the visual appearance of '{category}' ({definition}) in this image in **exactly one sentence**.",

    # P8: KB 참조 + 명시적 grounding 제약
    # 전략: "only what you see"로 이미지 외 정보 출력을 차단
    "P8": "Reference: {category} — {definition}. "
          "Describe only what you actually see in this image in **exactly one sentence**. "
          "Do **not** include details from the reference that are not visible.",

    # P9: KB를 용어 가이드로 제공 + 시각 요소 나열
    # 전략: KB를 "naming guide"로 명시하여 복사 방지, 관찰 가능한 요소만 나열
    "P9": "Use the following as a naming guide: {category} — {definition}. "
          "List the observable visual elements in this image in **exactly one sentence**, "
          "using the correct Korean cultural terms where applicable.",

    # P10: KB 조건부 매칭 — 보이는 것만 KB 용어로 연결
    # 전략: KB 특징 중 실제로 보이는 것만 선택적으로 사용하도록 지시
    "P10": "Given that {category} is {definition}, "
           "describe **only** the features from this definition that are actually visible in the image in **exactly one sentence**.",

    # P11: 이미지 우선 — KB는 보조
    # 전략: 이미지 묘사를 먼저 하고, KB 용어는 해당될 때만 사용
    "P11": "Describe what you see in this image, identifying elements that relate to '{category}' ({definition}) "
           "in **exactly one sentence**. **Only** mention cultural terms if visually supported.",

    # P12: 간결한 통합 — 시각 + 문화 용어 결합
    # 전략: P7의 단순함 + grounding 한 줄 제약
    "P12": "Describe the visual appearance of '{category}' ({definition}) shown in this image in **exactly one sentence**. "
           "**Only** describe what is directly observable.",

    # P13: P8 강화 — 용어 사용 명시 장려
    # 전략: P8 원본 + "use the correct Korean cultural term" 추가로 CoVT의 용어 치환 약점 보완
    "P13": "Reference: {category} — {definition}. "
           "Describe only what you actually see in this image in **exactly one sentence**, "
           "using the correct Korean cultural term where visually supported. "
           "Do **not** include details from the reference that are not visible.",

# ── KB 주입 프롬프트 v3 (P8/P11/P13 장점 결합) ──

    # P14: P11 이미지 우선 + P8 KB 격리 + P13 용어 장려
    # 전략: P11의 이미지 우선 순서로 환각 억제 + P8의 KB 격리로 복사 방지 + P13의 용어 장려
    # P11 재평가에서 환각 자연 억제 효과가 확인됨 (실질 동점), KB 격리를 추가하여 강화
    "P14": "Reference: {category} — {definition}. "
           "Describe what you see in this image in **exactly one sentence**, "
           "using the correct Korean cultural term from the reference where visually confirmed. "
           "Do **not** add anything from the reference that is not visible.",

    # P15: P13 + 시각 디테일 밀도 강화
    # 전략: P13 기반 + 색상/형태/배치 등 시각 속성 명시 요청
    # CoVT가 Base보다 캡션이 9~19자 짧고 시각 디테일 누락하는 문제 해결 
    "P15": "Reference: {category} — {definition}. "
           "Describe what you actually see in this image in **exactly one sentence**, "
           "noting colors, shapes, and spatial arrangement. "
           "Use the correct Korean cultural term if visually supported. "
           "Do **not** include any reference details that are not visible.",

    # P16: P8 KB 격리 강화 (naming guide) + P13 용어 장려 + P8 이중 grounding
    # 전략: KB를 "naming guide only"로 격리하여 정의 복사를 더 강하게 억제
    # P9의 naming guide 장점 + P8의 환각 차단 + P13의 용어 장려를 결합
    "P16": "Reference (naming guide only): {category} — {definition}. "
           "Describe only what you actually see in this image in **exactly one sentence**. "
           "Use the correct Korean cultural terms from the reference where visually supported, "
           "but do **not** include any details from the reference that are not visible.",

    # ── KB 주입 프롬프트 v4 (P13/P15/P16 장단점 교차 보완) ──

    # P17: P13 기반 + P15의 시각 디테일 밀도 보강
    # 전략: P13의 KB 용어+grounding 균형 유지 + "key visual details" 한 구 추가로 시각 밀도 향상
    # P15처럼 "colors, shapes, spatial arrangement" 전체 나열 시 KB 용어가 밀려나므로 압축
    "P17": "Reference: {category} — {definition}. "
           "Describe only what you actually see in this image in **exactly one sentence**, "
           "including key visual details such as colors and forms. "
           "Use the correct Korean cultural term where visually supported. "
           "Do **not** include details from the reference that are not visible.",

    # P18: P15 기반 + KB 용어 명시 요구 강화
    # 전략: P15의 시각 묘사 강점 유지 + "if visually supported" → "Name the subject using"으로 교체
    # P15에서 KB 용어 누락 / 카테고리명을 주어로 쓰는 이상 구조 방지
    "P18": "Reference: {category} — {definition}. "
           "Describe what you actually see in this image in **exactly one sentence**, "
           "noting colors, shapes, and spatial arrangement. "
           "Name the subject using the correct Korean cultural term from the reference. "
           "Do **not** include any reference details that are not visible.",

    # !! P19: P16 기반 + 격리 완화
    # 전략: "naming guide only" → "naming guide" (only 제거) + "Name and describe"로 용어 사용 적극 유도
    # P16에서 성묘/행궁/한빛탑 등 고유명사 누락 문제 해결
    "P19": "Reference (naming guide): {category} — {definition}. "
           "Name and describe only what you actually see in this image in **exactly one sentence**. "
           "Use the correct Korean cultural terms from the reference where visually supported, "
           "but do **not** include any details from the reference that are not visible.",

}


IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

# img/ 서브폴더 구조를 위한 카테고리 추출 함수
# 이미지 경로: .../Korean/{id}_{Category}/img/{filename}
# → 부모의 부모 디렉토리명에서 카테고리 추출
def extract_category_from_path(image_path: str) -> str:
    """
    img/ 서브폴더 구조에서 카테고리명 추출.
    예: '.../9025_Tteokguk_Mandu soup/img/xxx.jpg' -> 'Tteokguk and Mandu soup'
    예: '.../9008_Gyeongbokgung palace/img/xxx.jpg' -> 'Gyeongbokgung palace'
    """
    # 1. img/ 의 부모 디렉토리 (카테고리 폴더) 이름 가져오기
    img_dir = os.path.dirname(image_path)           # .../9001_Kimjang/img
    category_dir = os.path.basename(os.path.dirname(img_dir))  # 9001_Kimjang

    # 2. 맨 앞의 숫자_ 제거 (예: '9025_')
    clean_name = re.sub(r'^\d+_', '', category_dir)

    # 3. 남은 언더바(_)를 ' and '로 치환
    category = clean_name.replace('_', ' and ')

    return category


def load_model_and_processor(model_name: str, ckpt: Optional[str] = None):
    """
    Correct loader for Qwen2.5-VL and CoVT models.
    CoVT uses external visual features (SAM / Depth / DINO) via processor + prompt,
    NOT via additional model parameters. Warning about unused weights is expected.
    """

    src = ckpt if ckpt is not None else model_name
    print(f"[INFO] Loading model from: {src}")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        src,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,   # ⭐ 반드시 필요
    ).eval()

    processor = AutoProcessor.from_pretrained(
        src,
        trust_remote_code=True,
        min_pixels=256 * 28 * 28,
        max_pixels=384 * 28 * 28,   # OOM 방지용
    )

    ip = getattr(processor, "image_processor", None)
    print(
        "[PIXELS]",
        "min_pixels:", getattr(ip, "min_pixels", None),
        "max_pixels:", getattr(ip, "max_pixels", None),
    )

    print(
        "[INFO] NOTE: Warnings about unused SAM / Depth / DINO weights are EXPECTED "
        "for CoVT. Visual reasoning is injected via processor + prompt."
    )

    return model, processor


def extract_caption(text: str) -> str:
    # 1) <answer>...</answer>가 있으면 그 안의 내용만 추출
    m = re.search(r"<answer>(.*?)</answer>", text, flags=re.DOTALL)
    if m:
        text = m.group(1).strip()
    else:
        # 2) <think>...</think> 제거
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    # 3) 특수 토큰 및 잔여 태그 제거
    text = re.sub(r"<\|.*?\|>|<answer>|</answer>", "", text)
    text = re.sub(r"\n", " ", text)

    return text.strip()


def resize_if_needed(img: Image.Image, max_side: int = 1024) -> Image.Image:
    w, h = img.size
    if max(w, h) <= max_side:
        return img
    scale = max_side / max(w, h)
    nw, nh = int(w * scale), int(h * scale)
    return img.resize((nw, nh), Image.BICUBIC)


def run_single_inference(
    model,
    processor,
    image_path: str,
    question: str,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    top_p: float = 0.9,
    seed: int = 42,
):
    pil_image = Image.open(image_path).convert("RGB")
    pil_image = resize_if_needed(pil_image, max_side=1024)

    image_ref = image_path

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_ref},
                {"type": "text", "text": question},
            ],
        }
    ]

    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = processor(text=[prompt], images=[pil_image], return_tensors="pt")

    device = model.device
    inputs = {
        k: (v.to(device) if isinstance(v, torch.Tensor) else v)
        for k, v in inputs.items()
    }

    seed = int(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    start = time.time()
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=float(temperature),
            top_p=float(top_p),
            do_sample=(float(temperature) > 0.0),
            pad_token_id=processor.tokenizer.eos_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
        )
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - start

    input_len = inputs["input_ids"].shape[1]
    new_tokens = generated_ids[0, input_len:]
    raw = processor.decode(new_tokens, skip_special_tokens=True)
    answer = extract_caption(raw)

    return answer, elapsed


def append_jsonl(path: str, obj: Dict[str, Any]):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def list_images_in_img_dir(img_dir: str, limit: int = 40) -> List[str]:
    """
    img/ 디렉토리에서 이미지를 정렬 후 상위 limit개만 반환.
    """
    paths = []
    if not os.path.isdir(img_dir):
        return paths
    for fn in os.listdir(img_dir):
        ext = os.path.splitext(fn)[1].lower()
        if ext in IMG_EXTS:
            paths.append(os.path.join(img_dir, fn))
    paths.sort()
    return paths[:limit]


def collect_images(
    data_root: str,
    images_per_category: int = 50,
    category_filter: List[str] = None,
) -> List[Tuple[str, str]]:
    """
    data_root 아래 모든 카테고리 디렉토리의 img/ 폴더에서
    정렬된 순서로 상위 images_per_category개 이미지를 수집.

    category_filter: 카테고리 ID 접두사 목록 (예: ["9008", "9027"]).
                     None이면 전체 카테고리 수집.

    Returns: List of (image_id, image_path)
    """
    items: List[Tuple[str, str]] = []

    category_dirs = sorted([
        d for d in os.listdir(data_root)
        if os.path.isdir(os.path.join(data_root, d))
    ])

    if category_filter:
        category_dirs = [
            d for d in category_dirs
            if any(d.startswith(f) for f in category_filter)
        ]

    for cat_dir in category_dirs:
        img_dir = os.path.join(data_root, cat_dir, "img")
        paths = list_images_in_img_dir(img_dir, limit=images_per_category)
        if not paths:
            print(f"[WARN] No images found in: {img_dir}")
            continue
        for p in paths:
            image_id = os.path.splitext(os.path.basename(p))[0]
            items.append((image_id, p))

    return items


def load_done_ids(jsonl_path: str) -> set:
    done = set()
    if not os.path.exists(jsonl_path):
        return done
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                if "image_id" in obj:
                    done.add(obj["image_id"])
            except Exception:
                pass
    return done


def main():
    ap = argparse.ArgumentParser(
        description="Generate captions for the first N images per category "
                    "from KS-KEIT Korean dataset."
    )
    ap.add_argument(
        "--data_root",
        default="/home/qhdusrla08/projects/CoVT_senior/gradio/data/KS-KEIT_V3_Korean_0112/Korean",
        help="Root directory containing category subdirectories (each with an img/ folder)",
    )
    ap.add_argument("--images_per_category", type=int, default=9999,
                    help="Number of images to use per category (default: 40)")
    ap.add_argument("--out_dir", default="./outputs_korean", help="Output directory")
    ap.add_argument("--model_name", default="Qwen/Qwen2.5-VL-7B-Instruct",
                    help="HF model repo id")
    ap.add_argument("--ckpt", default=None, help="Local checkpoint path (optional)")
    ap.add_argument("--model_tag", default="base", help="Tag for output file naming")
    ap.add_argument("--prompt_id", default="P19", choices=list(PROMPTS.keys()))
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip_existing", action="store_true",
                    help="Skip images whose image_id already exists in the output file")
    ap.add_argument(
        "--categories", nargs="*", default=None,
        metavar="ID",
        help="카테고리 ID 접두사 필터 (예: 9008 9027 9028 9029). 미지정 시 전체 카테고리.",
    )
    args = ap.parse_args()

    print("[ARGS]", args)

    if not os.path.isdir(args.data_root):
        raise FileNotFoundError(f"data_root not found: {args.data_root}")

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"captions_{args.model_tag}_{args.prompt_id}.jsonl")

    prompt_template = PROMPTS[args.prompt_id]

    # 이미지 수집: 카테고리별 상위 N장
    items = collect_images(
        args.data_root,
        images_per_category=args.images_per_category,
        category_filter=args.categories,
    )
    n_cats = len(set(os.path.basename(os.path.dirname(os.path.dirname(p))) for _, p in items))
    print(f"[INFO] Total images collected: {len(items)} ({n_cats} categories)")
    if args.categories:
        print(f"[INFO] Category filter: {args.categories}")
    print(f"[INFO] Output: {out_path}")
    print(f"[INFO] Model: {args.model_name} (tag={args.model_tag})")
    print(f"[INFO] Prompt Template({args.prompt_id}): {prompt_template}")

    done = load_done_ids(out_path) if args.skip_existing else set()

    model, processor = load_model_and_processor(args.model_name, args.ckpt)

    n_ok, n_fail, n_skip = 0, 0, 0
    for idx, (image_id, image_path) in enumerate(items, 1):
        if args.skip_existing and image_id in done:
            n_skip += 1
            continue

        category_name = extract_category_from_path(image_path)

        if "{definition}" in prompt_template:
            definition = KNOWLEDGE_BASE.get(category_name, category_name)
            current_prompt = prompt_template.format(
                category=category_name, definition=definition
            )
        elif "{category}" in prompt_template:
            current_prompt = prompt_template.format(category=category_name)
        else:
            current_prompt = prompt_template

        rec: Dict[str, Any] = {
            "image_id": image_id,
            "category": category_name,
            "image_path": image_path,
            "model": args.model_tag,
            "prompt_id": args.prompt_id,
            "gen": {
                "temperature": args.temperature,
                "top_p": args.top_p,
                "max_new_tokens": args.max_new_tokens,
                "seed": args.seed,
            },
        }

        try:
            caption, elapsed = run_single_inference(
                model=model,
                processor=processor,
                image_path=image_path,
                question=current_prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                seed=args.seed,
            )
            rec["caption"] = caption
            rec["elapsed_sec"] = elapsed
            rec["status"] = "ok"
            n_ok += 1
        except Exception as e:
            rec["caption"] = ""
            rec["elapsed_sec"] = None
            rec["status"] = "fail"
            rec["error"] = repr(e)
            n_fail += 1

        append_jsonl(out_path, rec)

        if idx % 20 == 0:
            print(f"[INFO] {idx}/{len(items)} processed (ok={n_ok}, fail={n_fail}, skip={n_skip})")
            print(f"      - Current Category: {category_name}")

    print(f"[DONE] ok={n_ok}, fail={n_fail}, skip={n_skip}")
    print(f"[DONE] saved to: {out_path}")


if __name__ == "__main__":
    main()
