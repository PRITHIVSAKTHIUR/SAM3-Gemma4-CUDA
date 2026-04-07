import os
import json
import ast
import re
import cv2
import tempfile
import spaces
import gradio as gr
import numpy as np
import torch
import matplotlib
from PIL import Image, ImageDraw, ImageFont
from threading import Thread
from typing import Iterable

import supervision as sv

from transformers import (
    Sam3Model,
    Sam3Processor,
    Sam3VideoModel,
    Sam3VideoProcessor,
    Sam3TrackerModel,
    Sam3TrackerProcessor,
    Gemma4ForConditionalGeneration,
    AutoProcessor,
    TextIteratorStreamer,
)

from gradio.themes import Soft
from gradio.themes.utils import colors, fonts, sizes


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VL_DTYPE = (
    torch.bfloat16
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else (torch.float16 if torch.cuda.is_available() else torch.float32)
)

SAM_MODEL_NAME = "facebook/sam3"
VL_MODEL_NAME = "google/gemma-4-E2B-it"

MODEL_VL = "Gemma 4"

print(f"🖥️ Using compute device: {DEVICE}")
print("⏳ Loading models permanently into memory...")


colors.steel_blue = colors.Color(
    name="steel_blue",
    c50="#EBF3F8", c100="#D3E5F0", c200="#A8CCE1", c300="#7DB3D2",
    c400="#529AC3", c500="#4682B4", c600="#3E72A0", c700="#36638C",
    c800="#2E5378", c900="#264364", c950="#1E3450",
)


class SteelBlueTheme(Soft):
    def __init__(
        self,
        *,
        primary_hue: colors.Color | str = colors.gray,
        secondary_hue: colors.Color | str = colors.steel_blue,
        neutral_hue: colors.Color | str = colors.slate,
        text_size: sizes.Size | str = sizes.text_lg,
        font: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("Outfit"), "Arial", "sans-serif",
        ),
        font_mono: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace",
        ),
    ):
        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
            neutral_hue=neutral_hue,
            text_size=text_size,
            font=font,
            font_mono=font_mono,
        )
        super().set(
            background_fill_primary="*primary_50",
            background_fill_primary_dark="*primary_900",
            body_background_fill="linear-gradient(135deg, *primary_200, *primary_100)",
            body_background_fill_dark="linear-gradient(135deg, *primary_900, *primary_800)",
            button_primary_text_color="white",
            button_primary_text_color_hover="white",
            button_primary_background_fill="linear-gradient(90deg, *secondary_500, *secondary_600)",
            button_primary_background_fill_hover="linear-gradient(90deg, *secondary_600, *secondary_700)",
            button_primary_background_fill_dark="linear-gradient(90deg, *secondary_600, *secondary_800)",
            button_primary_background_fill_hover_dark="linear-gradient(90deg, *secondary_500, *secondary_500)",
            slider_color="*secondary_500",
            slider_color_dark="*secondary_600",
            block_title_text_weight="600",
            block_border_width="3px",
            block_shadow="*shadow_drop_lg",
            button_primary_shadow="*shadow_drop_lg",
            button_large_padding="11px",
            color_accent_soft="*primary_100",
            block_label_background_fill="*primary_200",
        )


steel_blue_theme = SteelBlueTheme()


css = r"""
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

body, .gradio-container { font-family: 'Outfit', sans-serif !important; }
footer { display: none !important; }

.app-header {
    background: linear-gradient(135deg, #1E3450 0%, #264364 30%, #3E72A0 70%, #4682B4 100%);
    border-radius: 16px; padding: 32px 40px; margin-bottom: 24px;
    position: relative; overflow: hidden;
    box-shadow: 0 8px 32px rgba(30,52,80,0.25);
}
.app-header::before {
    content:''; position:absolute; top:-50%; right:-20%;
    width:400px; height:400px;
    background:radial-gradient(circle,rgba(255,255,255,0.06) 0%,transparent 70%);
    border-radius:50%;
}
.header-content {
    display:flex; align-items:center; gap:24px;
    position:relative; z-index:1;
}
.header-icon-wrap {
    width:64px; height:64px; background:rgba(255,255,255,0.12);
    border-radius:16px; display:flex; align-items:center; justify-content:center;
    flex-shrink:0; backdrop-filter:blur(8px); border:1px solid rgba(255,255,255,0.15);
}
.header-icon-wrap svg {
    width:36px; height:36px;
    display:block;
}
.header-text h1 {
    font-size:2rem; font-weight:700; color:#fff;
    margin:0 0 8px 0; letter-spacing:-0.02em; line-height:1.2;
}
.header-meta { display:flex; align-items:center; gap:12px; flex-wrap:wrap; }
.meta-badge {
    display:inline-flex; align-items:center; gap:6px;
    background:rgba(255,255,255,0.12); color:rgba(255,255,255,0.9);
    padding:4px 12px; border-radius:20px;
    font-family:'IBM Plex Mono',monospace; font-size:0.8rem; font-weight:500;
    border:1px solid rgba(255,255,255,0.1); backdrop-filter:blur(4px);
}
.meta-badge svg {
    color:#ffffff !important;
    stroke:#ffffff !important;
}
.meta-sep {
    width:4px; height:4px; background:rgba(255,255,255,0.35);
    border-radius:50%; flex-shrink:0;
}
.meta-cap { color:rgba(255,255,255,0.65); font-size:0.85rem; font-weight:400; }

.tab-intro {
    display:flex; align-items:flex-start; gap:16px;
    background:linear-gradient(135deg,rgba(70,130,180,0.06),rgba(70,130,180,0.02));
    border:1px solid rgba(70,130,180,0.15); border-left:4px solid #4682B4;
    border-radius:10px; padding:18px 22px; margin-bottom:20px;
}
.dark .tab-intro {
    background:linear-gradient(135deg,rgba(70,130,180,0.1),rgba(70,130,180,0.04));
    border-color:rgba(70,130,180,0.25);
}
.intro-icon {
    width:40px; height:40px; background:rgba(70,130,180,0.1);
    border-radius:10px; display:flex; align-items:center; justify-content:center;
    flex-shrink:0; margin-top:2px;
}
.intro-icon svg { width:22px; height:22px; color:#4682B4; }
.intro-text { flex:1; }
.intro-text p { margin:0; color:#2E5378; font-size:0.95rem; line-height:1.6; }
.dark .intro-text p { color:#A8CCE1; }
.intro-text p.intro-sub { color:#64748b; font-size:0.85rem; margin-top:4px; }
.dark .intro-text p.intro-sub { color:#94a3b8; }

.section-heading {
    display:flex; align-items:center; gap:14px;
    margin:18px 0 14px 0; padding:0 2px;
}
.heading-icon {
    width:32px; height:32px;
    background:linear-gradient(135deg,#4682B4,#3E72A0);
    border-radius:8px; display:flex; align-items:center; justify-content:center;
    flex-shrink:0; box-shadow:0 2px 8px rgba(70,130,180,0.2);
}
.heading-icon svg { width:18px; height:18px; color:#fff; }
.heading-label {
    font-weight:600; font-size:1.05rem;
    color:#1E3450; letter-spacing:-0.01em;
}
.dark .heading-label { color:#D3E5F0; }
.heading-line {
    flex:1; height:1px;
    background:linear-gradient(90deg,rgba(70,130,180,0.2),transparent);
}

.status-indicator {
    display:flex; align-items:center; gap:10px;
    padding:10px 16px; margin-top:10px;
    background:rgba(70,130,180,0.04); border:1px solid rgba(70,130,180,0.12);
    border-radius:8px;
}
.status-dot {
    width:8px; height:8px; background:#22c55e;
    border-radius:50%; flex-shrink:0;
    animation:statusPulse 2s ease-in-out infinite;
}
@keyframes statusPulse {
    0%,100% { opacity:1; box-shadow:0 0 0 0 rgba(34,197,94,0.4); }
    50%     { opacity:0.7; box-shadow:0 0 0 4px rgba(34,197,94,0); }
}
.status-text { font-size:0.85rem; color:#64748b; font-style:italic; }

.card-label {
    display:flex; align-items:center; gap:8px;
    font-weight:600; font-size:0.8rem;
    text-transform:uppercase; letter-spacing:0.06em; color:#4682B4;
    margin-bottom:14px; padding-bottom:10px;
    border-bottom:1px solid rgba(70,130,180,0.1);
}
.card-label svg { width:16px; height:16px; }

.primary {
    border-radius:10px !important; font-weight:600 !important;
    letter-spacing:0.02em !important; transition:all 0.25s ease !important;
}
.primary:hover {
    transform:translateY(-2px) !important;
    box-shadow:0 6px 20px rgba(70,130,180,0.3) !important;
}

.gradio-textbox textarea {
    font-family:'IBM Plex Mono',monospace !important;
    font-size:0.92rem !important; line-height:1.7 !important;
    border-radius:8px !important;
}

label { font-weight:600 !important; }

.section-divider {
    height:1px; background:linear-gradient(90deg,transparent,rgba(70,130,180,0.2),transparent);
    margin:16px 0; border:none;
}

@media (max-width: 768px) {
    .app-header { padding: 20px 24px; }
    .header-text h1 { font-size: 1.5rem; }
    .header-content { flex-direction: column; align-items: flex-start; gap: 16px; }
}
"""


FIRE_LOGO_SVG = """
<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
  <path d="M13.5 2.5c.4 2.3-.4 4-1.7 5.6-1.3 1.6-2.6 3-2.6 5 0 1.8 1.2 3.2 2.8 3.2 1.8 0 3.1-1.4 3.1-3.5 0-1.1-.4-2.1-1.2-3.3 2.7 1.2 5.1 4.1 5.1 7.4 0 4-3.1 7.1-7.2 7.1-4.3 0-7.8-3.3-7.8-7.8 0-3.1 1.7-5.5 4-7.8 1.7-1.7 3.9-3.6 4.7-6z" fill="white"/>
</svg>
"""
SVG_IMAGE = '<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" d="m2.25 15.75 5.159-5.159a2.25 2.25 0 0 1 3.182 0l5.159 5.159m-1.5-1.5 1.409-1.409a2.25 2.25 0 0 1 3.182 0l2.909 2.909M3.75 21h16.5A2.25 2.25 0 0 0 22.5 18.75V5.25A2.25 2.25 0 0 0 20.25 3H3.75A2.25 2.25 0 0 0 1.5 5.25v13.5A2.25 2.25 0 0 0 3.75 21Z"/></svg>'
SVG_DETECT = '<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" d="M7.5 3.75H6A2.25 2.25 0 0 0 3.75 6v1.5M16.5 3.75H18A2.25 2.25 0 0 1 20.25 6v1.5m0 9V18A2.25 2.25 0 0 1 18 20.25h-1.5m-9 0H6A2.25 2.25 0 0 1 3.75 18v-1.5M15 12a3 3 0 1 1-6 0 3 3 0 0 1 6 0Z"/></svg>'
SVG_OUTPUT = '<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" d="M3.75 9.776c.112-.017.227-.026.344-.026h15.812c.117 0 .232.009.344.026m-16.5 0a2.25 2.25 0 0 0-1.883 2.542l.857 6a2.25 2.25 0 0 0 2.227 1.932H19.05a2.25 2.25 0 0 0 2.227-1.932l.857-6a2.25 2.25 0 0 0-1.883-2.542m-16.5 0V6A2.25 2.25 0 0 1 6 3.75h3.879a1.5 1.5 0 0 1 1.06.44l2.122 2.12a1.5 1.5 0 0 0 1.06.44H18A2.25 2.25 0 0 1 20.25 9v.776"/></svg>'
SVG_TEXT = '<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" d="M7.5 8.25h9m-9 3H12m-9.75 1.51c0 1.6 1.123 2.994 2.707 3.227 1.129.166 2.27.293 3.423.379.35.026.67.21.865.501L12 21l2.755-4.133a1.14 1.14 0 0 1 .865-.501 48.172 48.172 0 0 0 3.423-.379c1.584-.233 2.707-1.626 2.707-3.228V6.741c0-1.602-1.123-2.995-2.707-3.228A48.394 48.394 0 0 0 12 3c-2.392 0-4.744.175-7.043.513C3.373 3.746 2.25 5.14 2.25 6.741v6.018Z"/></svg>'
SVG_CHIP = '<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" d="M8.25 3v1.5M4.5 8.25H3m18 0h-1.5M4.5 12H3m18 0h-1.5m-15 3.75H3m18 0h-1.5M8.25 19.5V21M12 3v1.5m0 15V21m3.75-18v1.5m0 15V21m-9-1.5h10.5a2.25 2.25 0 0 0 2.25-2.25V6.75a2.25 2.25 0 0 0-2.25-2.25H6.75A2.25 2.25 0 0 0 4.5 6.75v10.5a2.25 2.25 0 0 0 2.25 2.25Z"/></svg>'
SVG_VIDEO = '<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" d="m15.75 10.5 4.72-2.36A.75.75 0 0 1 21.75 8.81v6.38a.75.75 0 0 1-1.28.67l-4.72-2.36m0-3v3m-10.5 6h9A2.25 2.25 0 0 0 16.5 17.25V6.75A2.25 2.25 0 0 0 14.25 4.5h-9A2.25 2.25 0 0 0 3 6.75v10.5A2.25 2.25 0 0 0 5.25 19.5Z"/></svg>'


try:
    print("   ... Loading SAM3 image model")
    SAM_MODEL = Sam3Model.from_pretrained(SAM_MODEL_NAME).to(DEVICE)
    SAM_PROCESSOR = Sam3Processor.from_pretrained(SAM_MODEL_NAME)

    print("   ... Loading SAM3 tracker model")
    TRK_MODEL = Sam3TrackerModel.from_pretrained(SAM_MODEL_NAME).to(DEVICE)
    TRK_PROCESSOR = Sam3TrackerProcessor.from_pretrained(SAM_MODEL_NAME)

    print("   ... Loading SAM3 video model")
    VID_MODEL = Sam3VideoModel.from_pretrained(SAM_MODEL_NAME).to(DEVICE, dtype=torch.bfloat16)
    VID_PROCESSOR = Sam3VideoProcessor.from_pretrained(SAM_MODEL_NAME)

    print("   ... Loading Gemma 4 model")
    VL_MODEL = Gemma4ForConditionalGeneration.from_pretrained(
        VL_MODEL_NAME,
        torch_dtype=VL_DTYPE,
        device_map="auto" if torch.cuda.is_available() else None,
    ).eval()
    if not torch.cuda.is_available():
        VL_MODEL = VL_MODEL.to(DEVICE)

    VL_PROCESSOR = AutoProcessor.from_pretrained(VL_MODEL_NAME)

    print("✅ All models loaded successfully!")

except Exception as e:
    print(f"❌ CRITICAL ERROR LOADING MODELS: {e}")
    SAM_MODEL = None
    SAM_PROCESSOR = None
    TRK_MODEL = None
    TRK_PROCESSOR = None
    VID_MODEL = None
    VID_PROCESSOR = None
    VL_MODEL = None
    VL_PROCESSOR = None


BRIGHT_YELLOW = sv.Color(r=255, g=230, b=0)
BLACK = sv.Color(r=0, g=0, b=0)
MASK_COLORS = [
    (255, 230, 0),
    (255, 99, 132),
    (54, 162, 235),
    (75, 192, 192),
    (153, 102, 255),
    (255, 159, 64),
]

VIDEO_COLORS_BGR = [
    (181, 120, 31),
    (13, 128, 255),
    (43, 161, 43),
    (41, 38, 214),
    (189, 102, 148),
    (74, 87, 140),
]


def safe_parse_json(text: str):
    text = text.strip()
    text = re.sub(r"^```(json)?", "", text)
    text = re.sub(r"```$", "", text)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    try:
        return ast.literal_eval(text)
    except Exception:
        return {}


def clamp_box_xyxy(box, width, height):
    x1, y1, x2, y2 = box
    x1 = max(0, min(width - 1, int(x1)))
    y1 = max(0, min(height - 1, int(y1)))
    x2 = max(0, min(width - 1, int(x2)))
    y2 = max(0, min(height - 1, int(y2)))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def build_vl_inputs(image: Image.Image, prompt_text: str):
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt_text},
        ]
    }]

    text = VL_PROCESSOR.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = VL_PROCESSOR(
        text=[text],
        images=[image],
        return_tensors="pt",
        padding=True
    )

    if torch.cuda.is_available():
        inputs = {k: v.to(VL_MODEL.device) if hasattr(v, "to") else v for k, v in inputs.items()}
    else:
        inputs = {k: v.to(DEVICE) if hasattr(v, "to") else v for k, v in inputs.items()}

    return inputs


def qwen_filter_regions(image: Image.Image, regions: list, user_prompt: str) -> dict:
    region_descriptions = []
    for idx, reg in enumerate(regions):
        x1, y1, x2, y2 = reg["bbox"]
        region_descriptions.append({
            "region_index": idx,
            "bbox": [x1, y1, x2, y2],
            "sam_score": round(float(reg["score"]), 4),
        })

    instruction = f"""
You are given an image and a list of candidate object regions proposed by a segmentation model.

User request:
"{user_prompt}"

Candidate regions:
{json.dumps(region_descriptions, indent=2)}

Task:
Select all candidate regions that match the user request.

Return ONLY valid JSON in this exact format:
{{
  "selected_region_indexes": [0, 2],
  "reason": "short explanation"
}}

Rules:
- Use only indexes from the candidate list.
- If nothing matches, return an empty list.
- Do not return markdown.
"""

    inputs = build_vl_inputs(image, instruction)

    with torch.inference_mode():
        gen_ids = VL_MODEL.generate(
            **inputs,
            max_new_tokens=512,
            use_cache=True,
            temperature=0.2,
            do_sample=False,
        )

    raw = VL_PROCESSOR.batch_decode(
        gen_ids[:, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )[0].strip()

    parsed = safe_parse_json(raw)
    if not isinstance(parsed, dict):
        parsed = {"selected_region_indexes": [], "reason": "Could not parse model output."}

    parsed.setdefault("selected_region_indexes", [])
    parsed.setdefault("reason", "")
    return parsed


def overlay_masks_on_image(base_image: Image.Image, masks: list, opacity: float = 0.45):
    base = base_image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))

    for i, mask in enumerate(masks):
        if isinstance(mask, torch.Tensor):
            mask = mask.detach().cpu().numpy()
        mask = np.array(mask).astype(np.uint8)

        if mask.ndim == 4:
            mask = mask[0]
        if mask.ndim == 3 and mask.shape[0] == 1:
            mask = mask[0]
        if mask.ndim == 3 and mask.shape[-1] == 1:
            mask = np.squeeze(mask, axis=-1)

        if mask.shape[::-1] != base.size:
            mask_pil = Image.fromarray((mask * 255).astype(np.uint8)).resize(base.size, Image.NEAREST)
        else:
            mask_pil = Image.fromarray((mask * 255).astype(np.uint8))

        color = MASK_COLORS[i % len(MASK_COLORS)]
        fill = Image.new("RGBA", base.size, color + (0,))
        alpha = mask_pil.point(lambda v: int(opacity * 255) if v > 0 else 0)
        fill.putalpha(alpha)
        overlay = Image.alpha_composite(overlay, fill)

    return Image.alpha_composite(base, overlay).convert("RGB")


def annotate_sam3_candidates(image: Image.Image, boxes: list, scores: list, masks: list):
    img = overlay_masks_on_image(image, masks, opacity=0.35)
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        color = MASK_COLORS[i % len(MASK_COLORS)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label = f"id={i} | {scores[i]:.2f}"
        tb = draw.textbbox((x1, max(0, y1 - 22)), label, font=font)
        draw.rectangle(tb, fill=color)
        draw.text((tb[0], tb[1]), label, fill="black", font=font)

    return img


def annotate_final_selection(image: Image.Image, selected_regions: list):
    if not selected_regions:
        return image.convert("RGB")

    img = overlay_masks_on_image(
        image,
        [item["mask"] for item in selected_regions],
        opacity=0.45
    )
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except Exception:
        font = ImageFont.load_default()

    for i, item in enumerate(selected_regions):
        x1, y1, x2, y2 = item["bbox"]
        draw.rectangle([x1, y1, x2, y2], outline=(255, 230, 0), width=4)
        label = f"{item['label']} | {item['score']:.2f}"
        tb = draw.textbbox((x1, max(0, y1 - 24)), label, font=font)
        draw.rectangle(tb, fill=(255, 230, 0))
        draw.text((tb[0], tb[1]), label, fill="black", font=font)

    return img


def format_json_output(selected_regions, vl_reason, original_prompt):
    return {
        "prompt": original_prompt,
        "num_selected": len(selected_regions),
        "selected_regions": [
            {
                "region_index": item["region_index"],
                "bbox": item["bbox"],
                "score": round(float(item["score"]), 4),
                "label": item["label"],
            }
            for item in selected_regions
        ],
        "vl_reason": vl_reason,
    }


def calc_timeout_duration(video_file, *args):
    return args[-1] if args else 60


def extract_boxes_from_masks(mask_data, width, height):
    boxes = []

    if mask_data is None:
        return boxes

    if isinstance(mask_data, torch.Tensor):
        mask_data = mask_data.detach().cpu().numpy()

    mask_data = np.array(mask_data)

    if mask_data.ndim == 4:
        mask_data = mask_data[0]
    if mask_data.ndim == 3 and mask_data.shape[0] == 1:
        mask_data = mask_data[0]

    if mask_data.ndim == 2:
        mask_data = np.expand_dims(mask_data, axis=0)

    if mask_data.ndim != 3:
        return boxes

    for single_mask in mask_data:
        single_mask = np.array(single_mask)
        if single_mask.shape[:2] != (height, width):
            single_mask = cv2.resize(
                single_mask.astype(np.float32),
                (width, height),
                interpolation=cv2.INTER_NEAREST
            )

        binary = single_mask > 0
        ys, xs = np.where(binary)
        if len(xs) == 0 or len(ys) == 0:
            boxes.append(None)
            continue

        x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
        boxes.append(clamp_box_xyxy([x1, y1, x2, y2], width, height))

    return boxes


def draw_video_masks_contours_and_boxes(frame_bgr, mask_data, prompt_text, scores=None):
    out = frame_bgr.copy()
    h, w = out.shape[:2]

    if mask_data is None:
        return out

    if isinstance(mask_data, torch.Tensor):
        mask_data = mask_data.detach().cpu().numpy()

    mask_data = np.array(mask_data)

    if mask_data.ndim == 4:
        mask_data = mask_data.squeeze(1)
    if mask_data.ndim == 2:
        mask_data = np.expand_dims(mask_data, axis=0)

    if mask_data.ndim != 3 or len(mask_data) == 0:
        return out

    boxes = extract_boxes_from_masks(mask_data, w, h)

    for i in range(len(mask_data)):
        color = VIDEO_COLORS_BGR[i % len(VIDEO_COLORS_BGR)]
        mask = mask_data[i]

        if mask.shape[:2] != (h, w):
            mask = cv2.resize(
                mask.astype(np.float32),
                (w, h),
                interpolation=cv2.INTER_NEAREST
            )

        binary = mask > 0
        if not np.any(binary):
            continue

        for c in range(3):
            out[:, :, c] = np.where(
                binary,
                (out[:, :, c].astype(np.float32) * 0.55 + color[c] * 0.45).astype(np.uint8),
                out[:, :, c],
            )

        contours, _ = cv2.findContours(
            binary.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(out, contours, -1, color, 2)

        box = boxes[i]
        if box is not None:
            x1, y1, x2, y2 = box
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

            if scores is not None and i < len(scores):
                try:
                    label = f"{prompt_text} {float(scores[i]):.2f}"
                except Exception:
                    label = f"{prompt_text} #{i}"
            else:
                label = f"{prompt_text} #{i}"

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            y_top = max(y1 - th - 10, 0)
            y_bottom = max(y1, th + 10)
            cv2.rectangle(out, (x1, y_top), (x1 + tw + 6, y_bottom), color, -1)
            cv2.putText(
                out,
                label,
                (x1 + 3, max(y1 - 4, th + 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

    return out


def apply_mask_overlay(base_image, mask_data, opacity=0.5):
    if isinstance(base_image, np.ndarray):
        base_image = Image.fromarray(base_image)
    base_image = base_image.convert("RGBA")

    if mask_data is None:
        return base_image.convert("RGB")

    if isinstance(mask_data, torch.Tensor):
        mask_data = mask_data.detach().cpu().numpy()
    mask_data = np.array(mask_data).astype(np.uint8)

    if mask_data.ndim == 4:
        mask_data = mask_data[0]
    if mask_data.ndim == 3 and mask_data.shape[0] == 1:
        mask_data = mask_data[0]

    if mask_data.ndim == 2:
        mask_data = [mask_data]
        num_masks = 1
    elif mask_data.ndim == 3:
        num_masks = mask_data.shape[0]
    else:
        return base_image.convert("RGB")

    try:
        color_map = matplotlib.colormaps["rainbow"].resampled(max(num_masks, 1))
    except AttributeError:
        import matplotlib.cm as cm
        color_map = cm.get_cmap("rainbow").resampled(max(num_masks, 1))

    rgb_colors = [tuple(int(c * 255) for c in color_map(i)[:3]) for i in range(num_masks)]
    composite_layer = Image.new("RGBA", base_image.size, (0, 0, 0, 0))

    for i, single_mask in enumerate(mask_data):
        mask_bitmap = Image.fromarray((single_mask * 255).astype(np.uint8))
        if mask_bitmap.size != base_image.size:
            mask_bitmap = mask_bitmap.resize(base_image.size, resample=Image.NEAREST)

        fill_color = rgb_colors[i]
        color_fill = Image.new("RGBA", base_image.size, fill_color + (0,))
        mask_alpha = mask_bitmap.point(lambda v: int(v * opacity) if v > 0 else 0)
        color_fill.putalpha(mask_alpha)
        composite_layer = Image.alpha_composite(composite_layer, color_fill)

    return Image.alpha_composite(base_image, composite_layer).convert("RGB")


def draw_points_on_image(image, points):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    draw_img = image.copy()
    draw = ImageDraw.Draw(draw_img)

    for pt in points:
        x, y = pt
        r = 8
        draw.ellipse((x - r, y - r, x + r, y + r), fill="red", outline="white", width=4)

    return draw_img


@spaces.GPU
def run_sam3_qwen_detection(image, prompt, conf_thresh):
    if SAM_MODEL is None or SAM_PROCESSOR is None or VL_MODEL is None or VL_PROCESSOR is None:
        raise gr.Error("Models failed to load on startup.")

    if image is None:
        raise gr.Error("Please upload an image.")
    if not prompt or not prompt.strip():
        raise gr.Error("Please provide a text prompt.")

    try:
        image = image.convert("RGB")

        model_inputs = SAM_PROCESSOR(
            images=image,
            text=prompt,
            return_tensors="pt"
        ).to(DEVICE)

        with torch.no_grad():
            sam_outputs = SAM_MODEL(**model_inputs)

        processed = SAM_PROCESSOR.post_process_instance_segmentation(
            sam_outputs,
            threshold=float(conf_thresh),
            mask_threshold=0.5,
            target_sizes=model_inputs.get("original_sizes").tolist()
        )[0]

        raw_masks = processed.get("masks", None)
        raw_scores = processed.get("scores", None)

        if raw_masks is None or raw_scores is None or len(raw_scores) == 0:
            empty_json = {
                "prompt": prompt,
                "num_selected": 0,
                "selected_regions": [],
                "vl_reason": "SAM3 found no candidate regions."
            }
            return image, image, json.dumps(empty_json, indent=2), "No detections found."

        raw_masks_np = raw_masks.detach().cpu().numpy()
        raw_scores_np = raw_scores.detach().cpu().numpy()

        h, w = image.size[1], image.size[0]
        candidate_regions = []

        for idx, mask in enumerate(raw_masks_np):
            if mask.ndim == 3:
                mask = np.squeeze(mask, axis=0)
            ys, xs = np.where(mask > 0)
            if len(xs) == 0 or len(ys) == 0:
                continue

            x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
            bbox = clamp_box_xyxy([x1, y1, x2, y2], w, h)

            candidate_regions.append({
                "region_index": len(candidate_regions),
                "bbox": bbox,
                "score": float(raw_scores_np[idx]),
                "mask": mask,
                "label": prompt,
            })

        if len(candidate_regions) == 0:
            empty_json = {
                "prompt": prompt,
                "num_selected": 0,
                "selected_regions": [],
                "vl_reason": "SAM3 masks were empty after post-processing."
            }
            return image, image, json.dumps(empty_json, indent=2), "No valid mask regions found."

        sam3_vis = annotate_sam3_candidates(
            image,
            [r["bbox"] for r in candidate_regions],
            [r["score"] for r in candidate_regions],
            [r["mask"] for r in candidate_regions],
        )

        vl_result = qwen_filter_regions(image, candidate_regions, prompt)
        selected_idx = vl_result.get("selected_region_indexes", [])
        reason = vl_result.get("reason", "")

        valid_idx = []
        for idx in selected_idx:
            try:
                idx = int(idx)
                if 0 <= idx < len(candidate_regions):
                    valid_idx.append(idx)
            except Exception:
                continue

        seen = set()
        valid_idx = [x for x in valid_idx if not (x in seen or seen.add(x))]

        selected_regions = [candidate_regions[i] for i in valid_idx]
        final_vis = annotate_final_selection(image, selected_regions)
        final_json = format_json_output(selected_regions, reason, prompt)

        status = (
            f"SAM3 proposed {len(candidate_regions)} region(s). "
            f"{MODEL_VL} selected {len(selected_regions)} region(s)."
        )

        return sam3_vis, final_vis, json.dumps(final_json, indent=2), status

    except Exception as e:
        raise gr.Error(f"Error during detection: {e}")


@spaces.GPU(duration=calc_timeout_duration)
def run_video_segmentation(video_path, prompt, frame_limit, time_limit):
    if VID_MODEL is None or VID_PROCESSOR is None:
        raise gr.Error("Video models failed to load on startup.")

    if not video_path:
        raise gr.Error("Please upload a video.")
    if not prompt or not prompt.strip():
        raise gr.Error("Please provide a text prompt.")

    try:
        video_cap = cv2.VideoCapture(video_path)
        vid_fps = video_cap.get(cv2.CAP_PROP_FPS)
        if not vid_fps or vid_fps <= 0:
            vid_fps = 24.0

        vid_w = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_h = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        video_frames = []
        counter = 0
        while video_cap.isOpened():
            ret, frame = video_cap.read()
            if not ret or (frame_limit > 0 and counter >= frame_limit):
                break
            video_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            counter += 1
        video_cap.release()

        if len(video_frames) == 0:
            return None, "No readable frames found in video."

        session = VID_PROCESSOR.init_video_session(
            video=video_frames,
            inference_device=DEVICE,
            dtype=torch.bfloat16
        )
        session = VID_PROCESSOR.add_text_prompt(
            inference_session=session,
            text=prompt
        )

        temp_out_path = tempfile.mktemp(suffix=".mp4")
        video_writer = cv2.VideoWriter(
            temp_out_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            vid_fps,
            (vid_w, vid_h)
        )

        processed_frames = 0
        annotated_frames = 0

        for model_out in VID_MODEL.propagate_in_video_iterator(
            inference_session=session,
            max_frame_num_to_track=len(video_frames)
        ):
            post_processed = VID_PROCESSOR.postprocess_outputs(session, model_out)
            f_idx = model_out.frame_idx

            frame_rgb = video_frames[f_idx]
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            if "masks" in post_processed and post_processed["masks"] is not None:
                detected_masks = post_processed["masks"]
                if hasattr(detected_masks, "ndim") and detected_masks.ndim == 4:
                    detected_masks = detected_masks.squeeze(1)

                scores = post_processed.get("scores", None)
                annotated_bgr = draw_video_masks_contours_and_boxes(
                    frame_bgr=frame_bgr,
                    mask_data=detected_masks,
                    prompt_text=prompt,
                    scores=scores,
                )
                if detected_masks is not None:
                    annotated_frames += 1
            else:
                annotated_bgr = frame_bgr

            video_writer.write(annotated_bgr)
            processed_frames += 1

        video_writer.release()

        return (
            temp_out_path,
            f"Video processing completed successfully. Processed {processed_frames} frame(s). "
            f"Annotated {annotated_frames} frame(s) with masks, contours, and bounding boxes."
        )

    except Exception as e:
        return None, f"Error during video processing: {str(e)}"


@spaces.GPU(duration=calc_timeout_duration)
def run_video_segmentation_mask(video_path, prompt, frame_limit, time_limit):
    if VID_MODEL is None or VID_PROCESSOR is None:
        raise gr.Error("Video models failed to load on startup.")

    if not video_path:
        raise gr.Error("Please upload a video.")
    if not prompt or not prompt.strip():
        raise gr.Error("Please provide a text prompt.")

    try:
        video_cap = cv2.VideoCapture(video_path)
        vid_fps = video_cap.get(cv2.CAP_PROP_FPS)
        if not vid_fps or vid_fps <= 0:
            vid_fps = 24.0

        vid_w = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_h = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        video_frames = []
        counter = 0
        while video_cap.isOpened():
            ret, frame = video_cap.read()
            if not ret or (frame_limit > 0 and counter >= frame_limit):
                break
            video_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            counter += 1
        video_cap.release()

        if len(video_frames) == 0:
            return None, "No readable frames found in video."

        session = VID_PROCESSOR.init_video_session(
            video=video_frames,
            inference_device=DEVICE,
            dtype=torch.bfloat16
        )
        session = VID_PROCESSOR.add_text_prompt(
            inference_session=session,
            text=prompt
        )

        temp_out_path = tempfile.mktemp(suffix=".mp4")
        video_writer = cv2.VideoWriter(
            temp_out_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            vid_fps,
            (vid_w, vid_h)
        )

        processed_frames = 0
        masked_frames = 0

        for model_out in VID_MODEL.propagate_in_video_iterator(
            inference_session=session,
            max_frame_num_to_track=len(video_frames)
        ):
            post_processed = VID_PROCESSOR.postprocess_outputs(session, model_out)
            f_idx = model_out.frame_idx

            original_pil = Image.fromarray(video_frames[f_idx])

            if "masks" in post_processed:
                detected_masks = post_processed["masks"]
                if hasattr(detected_masks, "ndim") and detected_masks.ndim == 4:
                    detected_masks = detected_masks.squeeze(1)

                final_frame = apply_mask_overlay(original_pil, detected_masks)
                masked_frames += 1
            else:
                final_frame = original_pil

            video_writer.write(cv2.cvtColor(np.array(final_frame), cv2.COLOR_RGB2BGR))
            processed_frames += 1

        video_writer.release()

        return (
            temp_out_path,
            f"Video mask processing completed successfully. Processed {processed_frames} frame(s). "
            f"Applied mask overlays to {masked_frames} frame(s)."
        )

    except Exception as e:
        return None, f"Error during video mask processing: {str(e)}"


@spaces.GPU
def run_image_click_gpu(input_image, x, y, points_state, labels_state):
    if TRK_MODEL is None or TRK_PROCESSOR is None:
        raise gr.Error("Tracker model failed to load.")

    if input_image is None:
        return input_image, [], []

    if points_state is None:
        points_state = []
    if labels_state is None:
        labels_state = []

    points_state.append([x, y])
    labels_state.append(1)

    try:
        input_points = [[points_state]]
        input_labels = [[labels_state]]

        inputs = TRK_PROCESSOR(
            images=input_image,
            input_points=input_points,
            input_labels=input_labels,
            return_tensors="pt"
        ).to(DEVICE)

        with torch.no_grad():
            outputs = TRK_MODEL(**inputs, multimask_output=False)

        masks = TRK_PROCESSOR.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"],
            binarize=True
        )[0]

        final_img = apply_mask_overlay(input_image, masks[0])
        final_img = draw_points_on_image(final_img, points_state)

        return final_img, points_state, labels_state

    except Exception as e:
        print(f"Tracker Error: {e}")
        return input_image, points_state, labels_state


def image_click_handler(image, evt: gr.SelectData, points_state, labels_state):
    x, y = evt.index
    return run_image_click_gpu(image, x, y, points_state, labels_state)


@spaces.GPU
def explain_detection(image, prompt, detection_json_text):
    if VL_MODEL is None or VL_PROCESSOR is None:
        raise gr.Error(f"{MODEL_VL} model failed to load.")
    if image is None:
        raise gr.Error("Please upload an image.")
    if not detection_json_text or not detection_json_text.strip():
        raise gr.Error("Run detection first.")

    image = image.convert("RGB")
    explain_prompt = f"""
You are given an image, the original user prompt, and a JSON detection result.

Original user prompt:
{prompt}

Detection JSON:
{detection_json_text}

Explain briefly:
1. What object(s) were selected
2. Why they match the prompt
3. Whether the result seems reliable

Keep the answer concise and readable.
"""

    inputs = build_vl_inputs(image, explain_prompt)

    streamer = TextIteratorStreamer(
        VL_PROCESSOR.tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
        timeout=120
    )

    thread = Thread(
        target=VL_MODEL.generate,
        kwargs=dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=512,
            use_cache=True,
            temperature=0.6,
            do_sample=True,
        )
    )
    thread.start()

    full_text = ""
    for token in streamer:
        full_text += token
        yield full_text

    thread.join()


def html_header():
    return f"""
    <div class="app-header">
        <div class="header-content">
            <div class="header-icon-wrap">{FIRE_LOGO_SVG}</div>
            <div class="header-text">
                <h1>SAM3 + Gemma 4 — Image & Video Segmentation</h1>
                <div class="header-meta">
                    <span class="meta-badge">{SVG_CHIP} {SAM_MODEL_NAME}</span>
                    <span class="meta-sep"></span>
                    <span class="meta-cap">SAM3 Proposals</span>
                    <span class="meta-sep"></span>
                    <span class="meta-cap">Gemma 4 Filtering</span>
                    <span class="meta-sep"></span>
                    <span class="meta-cap">Image + Video Segmentation </span>
                </div>
            </div>
        </div>
    </div>
    """


def html_tab_intro(icon_svg, title, description, detail=""):
    sub = f'<p class="intro-sub">{detail}</p>' if detail else ""
    return f"""
    <div class="tab-intro">
        <div class="intro-icon">{icon_svg}</div>
        <div class="intro-text">
            <p><strong>{title}</strong> &mdash; {description}</p>
            {sub}
        </div>
    </div>
    """


def html_section_heading(icon_svg, label):
    return f"""
    <div class="section-heading">
        <div class="heading-icon">{icon_svg}</div>
        <span class="heading-label">{label}</span>
        <div class="heading-line"></div>
    </div>
    """


def html_card_label(icon_svg, label):
    return f'<div class="card-label">{icon_svg}<span>{label}</span></div>'


def html_status_indicator(text):
    return f"""
    <div class="status-indicator">
        <span class="status-dot"></span>
        <span class="status-text">{text}</span>
    </div>
    """


def html_divider():
    return '<div class="section-divider"></div>'


EXAMPLES = [
    ["examples/1.jpg", "grapes", 0.45],
    ["examples/2.jpg", "face", 0.35],
    ["examples/3.jpg", "croissant", 0.30],
]

VIDEO_EXAMPLES = [
    ["examples/1V.mp4", "cheetah", 120, 120],
]


with gr.Blocks() as demo:
    gr.HTML(html_header())

    with gr.Tabs():
        with gr.Tab("Image Detection (*Filter)"):
            gr.HTML(html_tab_intro(
                SVG_IMAGE,
                "Image Detection with SAM3 + Gemma 4",
                "SAM3 first proposes candidate masks and regions from your text prompt. Gemma 4 then filters those candidates and keeps only the regions that best match the request.",
                "Image mode: SAM3 proposes regions, Gemma 4 filters final detections.",
            ))

            with gr.Row():
                with gr.Column(scale=1):
                    gr.HTML(html_card_label(SVG_IMAGE, "Input"))
                    image_input = gr.Image(type="pil", label="Upload Image", height=360)

                    prompt_input = gr.Textbox(
                        label="Detection Prompt",
                        placeholder="e.g., person wearing a black top",
                        lines=2,
                    )

                    with gr.Accordion("Advanced Settings", open=False):
                        conf_slider = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.45,
                            step=0.05,
                            label="SAM3 Confidence Threshold",
                        )

                    detect_btn = gr.Button("Run SAM3 + Gemma 4 Detection", variant="primary")
                    explain_btn = gr.Button("Explain Result", variant="secondary")

                    gr.HTML(html_divider())

                    gr.Examples(
                        examples=EXAMPLES,
                        inputs=[image_input, prompt_input, conf_slider],
                        label="Examples",
                    )

                with gr.Column(scale=1):
                    gr.HTML(html_section_heading(SVG_DETECT, "SAM3 Candidate Proposals"))
                    sam3_output = gr.Image(label="SAM3 Result", height=300)

                    gr.HTML(html_section_heading(SVG_OUTPUT, "Final Gemma-Filtered Detection"))
                    final_output = gr.Image(label="SAM3 + Gemma 4 Result", height=300)

                    gr.Markdown(
                        f"""
                        ### How to Use
                    
                        #### 1. Upload & Prompt
                        - Upload an image you want to analyze & Enter a clear detection prompt.
                    
                        #### 2. Adjust SAM3 Settings
                        - Use the **confidence threshold slider** to control how strict SAM3 is:
                          - **Lower values** → more regions, **Higher values** → fewer, cleaner regions
                    
                        #### 3. Run Detection & Explain
                        - Click **"Run SAM3 + Gemma 4 Detection"**
                        - **Top Panel:** Candidate regions, **Bottom Panel:** Final filtered detections (Gemma 4)
                        - **JSON Output:** Structured results including bounding boxes, scores, and labels
                        - Click **"Explain Result"** to get a natural language explanation
                        """
                    )

                with gr.Column(scale=1):
                    gr.HTML(html_section_heading(SVG_TEXT, "Structured Output"))
                    json_output = gr.Textbox(label="Detection JSON", lines=18, interactive=True)

                    status_output = gr.Textbox(label="System Status", interactive=False)

                    gr.HTML(html_status_indicator(
                        "Pipeline: SAM3 proposes regions → Gemma 4 filters relevant detections."
                    ))

                    gr.HTML(html_section_heading(SVG_TEXT, "Gemma Explanation"))
                    explanation_output = gr.Textbox(label="Explanation", lines=15, interactive=True)

            detect_btn.click(
                fn=run_sam3_qwen_detection,
                inputs=[image_input, prompt_input, conf_slider],
                outputs=[sam3_output, final_output, json_output, status_output],
            )

            explain_btn.click(
                fn=explain_detection,
                inputs=[image_input, prompt_input, json_output],
                outputs=[explanation_output],
            )

        with gr.Tab("Video Segmentation (*Mask)"):
            gr.HTML(html_tab_intro(
                SVG_VIDEO,
                "Video Segmentation with SAM3 Mask Overlay",
                "Segment objects across video frames using a text prompt and render pure colored mask overlays directly on the original frames.",
                "Video mode: text-prompted segmentation with mask overlays only.",
            ))

            with gr.Row():
                with gr.Column(scale=1):
                    gr.HTML(html_card_label(SVG_VIDEO, "Video Input"))
                    video_input_mask = gr.Video(label="Upload Video", format="mp4", height=320)

                    video_prompt_mask = gr.Textbox(
                        label="Segmentation Prompt",
                        placeholder="e.g., players, person running, red car",
                        lines=2,
                    )

                    with gr.Accordion("Advanced Settings", open=False):
                        with gr.Row():
                            frame_limiter_mask = gr.Slider(
                                minimum=10,
                                maximum=1000,
                                value=60,
                                step=10,
                                label="Max Frames",
                            )
                            time_limiter_mask = gr.Radio(
                                choices=[60, 120, 180, 240, 300],
                                value=60,
                                label="Timeout (seconds)",
                            )

                    video_btn_mask = gr.Button("Run Video Mask Segmentation", variant="primary")

                    gr.HTML(html_divider())

                    gr.Examples(
                        examples=VIDEO_EXAMPLES,
                        inputs=[video_input_mask, video_prompt_mask, frame_limiter_mask, time_limiter_mask],
                        label="Video Examples",
                    )

                with gr.Column(scale=1):
                    gr.HTML(html_section_heading(SVG_OUTPUT, "Processed Video"))
                    video_output_mask = gr.Video(label="Masked Video", height=420)

                    video_status_mask = gr.Textbox(label="System Status", interactive=False)

                    gr.HTML(html_status_indicator(
                        "Pipeline: SAM3 video session → prompt conditioning → mask propagation with overlay rendering."
                    ))

            video_btn_mask.click(
                fn=run_video_segmentation_mask,
                inputs=[video_input_mask, video_prompt_mask, frame_limiter_mask, time_limiter_mask],
                outputs=[video_output_mask, video_status_mask],
            )

        with gr.Tab("Video Segmentation (*Annotated)"):
            gr.HTML(html_tab_intro(
                SVG_VIDEO,
                "Video Segmentation with SAM3",
                "Segment objects across video frames using a text prompt. The SAM3 video model initializes a video session and propagates segmentation masks through the clip.",
                "Video mode: text-prompted segmentation over tracked frames with masks, contours, and bounding boxes.",
            ))

            with gr.Row():
                with gr.Column(scale=1):
                    gr.HTML(html_card_label(SVG_VIDEO, "Video Input"))
                    video_input = gr.Video(label="Upload Video", format="mp4", height=320)

                    video_prompt = gr.Textbox(
                        label="Segmentation Prompt",
                        placeholder="e.g., players, person running, red car",
                        lines=2,
                    )

                    with gr.Accordion("Advanced Settings", open=False):
                        with gr.Row():
                            frame_limiter = gr.Slider(
                                minimum=10,
                                maximum=1000,
                                value=60,
                                step=10,
                                label="Max Frames",
                            )
                            time_limiter = gr.Radio(
                                choices=[60, 120, 180, 240, 300],
                                value=60,
                                label="Timeout (seconds)",
                            )

                    video_btn = gr.Button("Run Video Segmentation", variant="primary")

                    gr.HTML(html_divider())

                    gr.Examples(
                        examples=VIDEO_EXAMPLES,
                        inputs=[video_input, video_prompt, frame_limiter, time_limiter],
                        label="Video Examples",
                    )

                with gr.Column(scale=1):
                    gr.HTML(html_section_heading(SVG_OUTPUT, "Processed Video"))
                    video_output = gr.Video(label="Segmented Video", height=420)

                    video_status = gr.Textbox(label="System Status", interactive=False)

                    gr.HTML(html_status_indicator(
                        "Pipeline: SAM3 video session → prompt conditioning → mask propagation with contours and bounding boxes."
                    ))

            video_btn.click(
                fn=run_video_segmentation,
                inputs=[video_input, video_prompt, frame_limiter, time_limiter],
                outputs=[video_output, video_status],
            )

        with gr.Tab("Image Click Segmentation"):
            gr.HTML(html_tab_intro(
                SVG_IMAGE,
                "Interactive Click Segmentation with SAM3 Tracker",
                "Upload an image and click on the object you want to segment. Each click is treated as a positive foreground point and the tracker model updates the mask preview.",
                "Interactive mode: click-to-segment with cumulative positive points.",
            ))

            with gr.Row():
                with gr.Column(scale=1):
                    gr.HTML(html_card_label(SVG_IMAGE, "Interactive Input"))
                    img_click_input = gr.Image(
                        type="pil",
                        label="Upload Image",
                        interactive=True,
                        height=450
                    )

                    with gr.Row():
                        img_click_clear = gr.Button("Clear Points & Reset", variant="primary")

                    st_click_points = gr.State([])
                    st_click_labels = gr.State([])

                with gr.Column(scale=1):
                    gr.HTML(html_section_heading(SVG_OUTPUT, "Result Preview"))
                    img_click_output = gr.Image(
                        type="pil",
                        label="Segmented Preview",
                        height=450,
                        interactive=False
                    )

                    gr.HTML(html_status_indicator(
                        "Pipeline: click points → SAM3 tracker prompt encoding → mask prediction overlay."
                    ))

            img_click_input.select(
                fn=image_click_handler,
                inputs=[img_click_input, st_click_points, st_click_labels],
                outputs=[img_click_output, st_click_points, st_click_labels]
            )

            img_click_clear.click(
                fn=lambda: (None, [], []),
                outputs=[img_click_output, st_click_points, st_click_labels]
            )


if __name__ == "__main__":
    demo.launch(
        css=css,
        mcp_server=True,
        theme=steel_blue_theme,
        show_error=True,
        ssr_mode=False,
    )