"""
Skin Cancer Scanner v10 — Multi-Model (ViT + ConvNeXt V2 + EfficientNet-B1 + DINOv2)
Supports 4 classification models with selectable model in the UI.
Models:
  1. ViT: bnmbanhmi/seekwell_skincancer_v2 (HuggingFace)
  2. ConvNeXt V2 Tiny: locally trained (best_weights.pth)
  3. EfficientNet-B1: locally trained (best_weights.pth)
  4. DINOv2: Jayanth2002/dinov2-base-finetuned-SkinDisease (HuggingFace, 31-class → 6-class mapping)
Classes: ACK, BCC, MEL, NEV, SCC, SEK
"""
import os, io, time, traceback, json
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from transformers import ViTForImageClassification, AutoModelForImageClassification, AutoImageProcessor
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from typing import Optional
import timm
from torchvision import transforms

# ─── Config ────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
PROJECT_DIR = BASE_DIR.parent
MODEL_HF_NAME = "bnmbanhmi/seekwell_skincancer_v2"

# ─── 6 คลาสที่โมเดลจำแนก (ตาม config.json id2label) ─────────────
MODEL_CLASSES = [
    "ACK",    # 0 — Actinic Keratosis (ไม่ใช้ — ถูกบังคับ logit = -inf)
    "BCC",    # 1 — Basal Cell Carcinoma
    "MEL",    # 2 — Melanoma
    "NEV",    # 3 — Nevi (ไฝ)
    "SCC",    # 4 — Squamous Cell Carcinoma
    "SEK",    # 5 — Seborrheic Keratosis
]

NUM_MODEL_CLASSES = 6

# คลาสที่ไม่ต้องการ (บังคับ logit = -inf ก่อน softmax)
EXCLUDED_CLASS_INDICES = [0]  # ACK (index 0)

# ─── 5 คลาสแสดงผล (ตัด ACK ออก) ──────────────────
DISPLAY_CLASSES = [
    "MEL",    # (1) Melanoma — มะเร็งเมลาโนมา
    "BCC",    # (2) Basal Cell Carcinoma — มะเร็งเบเซลเซลล์
    "SCC",    # (3) Squamous Cell Carcinoma — มะเร็งสความัสเซลล์
    "NEV",    # (4) Nevi — ไฝธรรมดา
    "SEK",    # (5) Seborrheic Keratosis — กระเนื้อ
]

NUM_DISPLAY_CLASSES = 5

# Mapping: model class → display class (ไม่มี ACK)
CLASS_MERGE_MAP = {
    "BCC": "BCC",
    "MEL": "MEL",
    "NEV": "NEV",
    "SCC": "SCC",
    "SEK": "SEK",
}

CLASS_INFO = {
    "MEL": {
        "name_th": "เมลาโนมา (Melanoma)",
        "name_en": "Melanoma",
        "risk": "critical",
        "color": "#c0392b",
        "icon": "🚨",
        "desc": "มะเร็งผิวหนังชนิดร้ายแรงที่สุด สามารถแพร่กระจายได้",
        "advice": "ควรพบแพทย์ผิวหนังทันที — อย่ารอ!",
        "recommendations": [
            "ลักษณะ: ไฝ/จุดดำที่มีรูปร่างไม่สมมาตร ขอบไม่เรียบ สีไม่สม่ำเสมอ ขนาดใหญ่ หรือเปลี่ยนแปลง",
            "ความอันตราย: สามารถแพร่กระจายไปต่อมน้ำเหลืองและอวัยวะอื่นได้ หากรักษาช้า",
            "การรักษา: ผ่าตัดออกกว้างเป็นหลัก อาจร่วมกับ ภูมิคุ้มกันบำบัด (Immunotherapy)",
            "การพยากรณ์โรค: หากพบและรักษาระยะแรก อัตรารอดชีวิต 5 ปี > 95%",
            "🚨 ต้องพบแพทย์ผิวหนังทันที เพื่อตัดชิ้นเนื้อตรวจวินิจฉัยและวางแผนรักษา"
        ]
    },
    "BCC": {
        "name_th": "มะเร็งเบเซลเซลล์ (BCC)",
        "name_en": "Basal Cell Carcinoma",
        "risk": "high",
        "color": "#e74c3c",
        "icon": "⚠️",
        "desc": "มะเร็งผิวหนังชนิดพบบ่อยที่สุด มักไม่แพร่กระจาย แต่ควรรักษาเร็ว",
        "advice": "ควรพบแพทย์ผิวหนังภายใน 2 สัปดาห์",
        "recommendations": [
            "ลักษณะ: ก้อนนูนใส คล้ายไข่มุก มีเส้นเลือดฝอย หรือแผลเรื้อรังที่ไม่หาย",
            "สาเหตุ: การได้รับแสง UV สะสมเป็นเวลานาน โดยเฉพาะคนผิวขาว",
            "การรักษา: ผ่าตัดตัดออก จี้ไฟฟ้า หรือครีมทาเฉพาะ (Imiquimod) ตามขนาดและตำแหน่ง",
            "การป้องกัน: ทาครีมกันแดด SPF50+ ทุกวัน หลีกเลี่ยงแดดจัด 10:00-16:00",
            "⚠️ ควรพบแพทย์ผิวหนังเพื่อตัดชิ้นเนื้อตรวจยืนยัน"
        ]
    },
    "SCC": {
        "name_th": "มะเร็งสความัสเซลล์ (SCC)",
        "name_en": "Squamous Cell Carcinoma",
        "risk": "high",
        "color": "#d35400",
        "icon": "⚠️",
        "desc": "มะเร็งผิวหนังที่พบบ่อยเป็นอันดับ 2 สามารถแพร่กระจายได้หากไม่รักษา",
        "advice": "ควรพบแพทย์ผิวหนังภายใน 2 สัปดาห์",
        "recommendations": [
            "ลักษณะ: ก้อนนูนแดง ผิวหยาบเป็นสะเก็ด แผลเรื้อรังที่ไม่หาย มักเกิดบริเวณที่โดนแดด",
            "สาเหตุ: รังสี UV สะสม ภูมิคุ้มกันต่ำ แผลเรื้อรัง หรือพัฒนาจาก Actinic Keratosis",
            "ความเสี่ยง: สามารถแพร่กระจายไปต่อมน้ำเหลืองได้ถ้าไม่รักษา (2-5%)",
            "การรักษา: ผ่าตัดตัดออกเป็นหลัก อาจร่วมกับรังสีรักษา",
            "⚠️ ควรพบแพทย์ผิวหนังโดยเร็วเพื่อตัดชิ้นเนื้อตรวจยืนยัน"
        ]
    },
    "NEV": {
        "name_th": "ไฝ/ขี้แมลงวัน (Nevi)",
        "name_en": "Melanocytic Nevi",
        "risk": "low",
        "color": "#27ae60",
        "icon": "✅",
        "desc": "ไฝธรรมดา ไม่เป็นอันตราย แต่ควรสังเกตการเปลี่ยนแปลง",
        "advice": "ติดตามสังเกตทุก 6 เดือน ถ้ามีการเปลี่ยนแปลงควรพบแพทย์",
        "recommendations": [
            "ลักษณะ: จุดสีน้ำตาล/ดำ ขอบเรียบ สีสม่ำเสมอ ขนาด < 6 มม.",
            "การสังเกต ABCDE: A=ไม่สมมาตร B=ขอบไม่เรียบ C=สีไม่สม่ำเสมอ D=ขนาด>6มม. E=เปลี่ยนแปลง",
            "การดูแล: ถ่ายภาพเก็บไว้เปรียบเทียบทุก 3-6 เดือน",
            "การป้องกัน: ทาครีมกันแดด หลีกเลี่ยงแดดจัด ไม่แกะเกา",
            "เมื่อไรต้องพบแพทย์: ถ้าเข้าเกณฑ์ ABCDE อย่างใดอย่างหนึ่ง"
        ]
    },
    "SEK": {
        "name_th": "กระเนื้อ (Seborrheic Keratosis)",
        "name_en": "Seborrheic Keratosis",
        "risk": "low",
        "color": "#3498db",
        "icon": "✅",
        "desc": "รอยโรคผิวหนังชนิดไม่ร้ายแรง เช่น กระเนื้อ พบบ่อยในผู้สูงอายุ",
        "advice": "ไม่เป็นอันตราย ไม่ต้องรักษาถ้าไม่มีอาการ",
        "recommendations": [
            "ลักษณะ: ก้อนนูนสีน้ำตาล/ดำ ผิวขรุขระคล้ายขี้ผึ้ง ติดแน่นบนผิว",
            "สาเหตุ: เกิดจากการเจริญเติบโตของเซลล์ผิวหนังตามวัย (>40 ปี)",
            "การดูแล: ไม่ต้องรักษา เป็นเรื่องปกติ",
            "การรักษา: ถ้ารบกวน สามารถจี้เย็น จี้ไฟฟ้า หรือขูดออกได้",
            "เมื่อไรต้องพบแพทย์: ถ้าเปลี่ยนสี โตเร็ว หรือเจ็บ"
        ]
    },
}


# ─── Lazy Loading Models ────────────────────────────────────
# โหลดโมเดลเฉพาะตอนใช้งานจริง เพื่อประหยัด RAM (HF cpu-basic = 2GB)
print("=" * 65)
print("  Skin Cancer Scanner v10 — Multi-Model (Lazy Loading)")
print(f"  5 Display Classes | 4 Models (ViT, ConvNeXt V2, EfficientNet-B1, DINOv2)")
print("=" * 65)

# Device
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("  Device: Apple MPS (GPU)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("  Device: CUDA GPU")
else:
    device = torch.device("cpu")
    print("  Device: CPU")

# ─── Standard image transform for timm models (224x224) ───
timm_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ─── Lazy model cache ───
_model_cache = {}

CONVNEXT_HF_NAME = "conan17970/convnextv2-skin-cancer-isic2019"
EFFICIENTNET_HF_NAME = "conan17970/efficientnet-b1-skin-cancer-isic2019"
DINOV2_HF_NAME = "Jayanth2002/dinov2-base-finetuned-SkinDisease"

def _get_model(model_key: str):
    """Lazy load a model only when needed. Returns (model, processor_or_none)."""
    global _model_cache
    if model_key in _model_cache:
        return _model_cache[model_key]

    print(f"  [Lazy] Loading model: {model_key}...")
    result = (None, None)

    try:
        if model_key == "vit":
            m = ViTForImageClassification.from_pretrained(MODEL_HF_NAME)
            p = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
            m.eval(); m = m.to(device)
            result = (m, p)
            print(f"  [Lazy] ViT loaded!")

        elif model_key == "convnext":
            from huggingface_hub import hf_hub_download
            w = hf_hub_download(repo_id=CONVNEXT_HF_NAME, filename="best_weights.pth")
            m = timm.create_model("convnextv2_tiny.fcmae_ft_in22k_in1k", pretrained=False, num_classes=NUM_MODEL_CLASSES)
            m.load_state_dict(torch.load(w, map_location="cpu", weights_only=True))
            m.eval(); m = m.to(device)
            result = (m, None)
            print(f"  [Lazy] ConvNeXt V2 loaded!")

        elif model_key == "efficientnet":
            from huggingface_hub import hf_hub_download
            w = hf_hub_download(repo_id=EFFICIENTNET_HF_NAME, filename="best_weights.pth")
            m = timm.create_model("efficientnet_b1", pretrained=False, num_classes=NUM_MODEL_CLASSES)
            m.load_state_dict(torch.load(w, map_location="cpu", weights_only=True))
            m.eval(); m = m.to(device)
            result = (m, None)
            print(f"  [Lazy] EfficientNet-B1 loaded!")

        elif model_key == "dinov2":
            m = AutoModelForImageClassification.from_pretrained(DINOV2_HF_NAME)
            p = AutoImageProcessor.from_pretrained(DINOV2_HF_NAME)
            m.eval(); m = m.to(device)
            result = (m, p)
            print(f"  [Lazy] DINOv2 loaded!")

    except Exception as e:
        print(f"  [Lazy] {model_key} load error: {e}")
        result = (None, None)

    _model_cache[model_key] = result
    return result

# DINOv2 31 classes → our 6 classes mapping
DINOV2_31_CLASSES = [
    'Basal Cell Carcinoma', 'Darier_s Disease', 'Epidermolysis Bullosa Pruriginosa',
    'Hailey-Hailey Disease', 'Herpes Simplex', 'Impetigo', 'Larva Migrans',
    'Leprosy Borderline', 'Leprosy Lepromatous', 'Leprosy Tuberculoid',
    'Lichen Planus', 'Lupus Erythematosus Chronicus Discoides', 'Melanoma',
    'Molluscum Contagiosum', 'Mycosis Fungoides', 'Neurofibromatosis',
    'Papilomatosis Confluentes And Reticulate', 'Pediculosis Capitis',
    'Pityriasis Rosea', 'Porokeratosis Actinic', 'Psoriasis', 'Tinea Corporis',
    'Tinea Nigra', 'Tungiasis', 'actinic keratosis', 'dermatofibroma',
    'nevus', 'pigmented benign keratosis', 'seborrheic keratosis',
    'squamous cell carcinoma', 'vascular lesion',
]

DINOV2_TO_OUR_MAP = {
    0: 1, 12: 2, 19: 0, 24: 0, 26: 3, 27: 5, 28: 5, 29: 4,
}

# ─── Model Registry (all active=True, loaded lazily) ───
MODEL_REGISTRY = {
    "vit": {
        "name": "ViT (Vision Transformer)",
        "name_short": "ViT",
        "description": "Vision Transformer fine-tuned for skin cancer",
        "source": MODEL_HF_NAME,
        "accuracy": "HuggingFace Pre-trained",
        "active": True,
        "icon": "🔬",
    },
    "convnext": {
        "name": "ConvNeXt V2 Tiny",
        "name_short": "ConvNeXt V2",
        "description": "ConvNeXt V2 Tiny trained on ISIC 2019",
        "source": CONVNEXT_HF_NAME,
        "accuracy": "F1: 0.747 | Acc: 73.2%",
        "active": True,
        "icon": "🧠",
    },
    "efficientnet": {
        "name": "EfficientNet-B1",
        "name_short": "EfficientNet",
        "description": "EfficientNet-B1 trained on ISIC 2019",
        "source": EFFICIENTNET_HF_NAME,
        "accuracy": "F1: 0.688 | Acc: 68.2%",
        "active": True,
        "icon": "⚡",
    },
    "dinov2": {
        "name": "DINOv2 (31-class)",
        "name_short": "DINOv2",
        "description": "DINOv2 fine-tuned SkinDisease 31 classes → 6 classes",
        "source": DINOV2_HF_NAME,
        "accuracy": "Acc: 95.6% (31-class)",
        "active": True,
        "icon": "🦕",
    },
}

print(f"\n{'=' * 65}")
print(f"  4 Models registered (lazy loading — loaded on first use)")
for key, info in MODEL_REGISTRY.items():
    print(f"     {info['name']}")
print(f"  Output: {NUM_MODEL_CLASSES} classes -> {NUM_DISPLAY_CLASSES} display groups")
print(f"{'=' * 65}")


# ─── Model Prediction Functions ────────────────────────────

def _predict_vit(img: Image.Image) -> np.ndarray:
    """Run ViT model, return raw logits (6 classes)"""
    vit_model, vit_processor = _get_model("vit")
    if vit_model is None:
        return np.zeros(NUM_MODEL_CLASSES)
    encoding = vit_processor(img, return_tensors="pt")
    encoding = {k: v.to(device) for k, v in encoding.items()}
    with torch.no_grad():
        outputs = vit_model(**encoding)
        logits = outputs.logits.cpu().numpy()[0]
    return logits


def _predict_timm(img: Image.Image, model_key: str) -> np.ndarray:
    """Run a timm model (ConvNeXt V2 or EfficientNet-B1), return raw logits (6 classes)"""
    model, _ = _get_model(model_key)
    if model is None:
        return np.zeros(NUM_MODEL_CLASSES)
    input_tensor = timm_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(input_tensor).cpu().numpy()[0]
    return logits


def _predict_dinov2(img: Image.Image) -> np.ndarray:
    """Run DINOv2 (31-class) and map to our 6 classes by summing probabilities."""
    dinov2_model, dinov2_processor = _get_model("dinov2")
    if dinov2_model is None:
        return np.zeros(NUM_MODEL_CLASSES)
    encoding = dinov2_processor(img, return_tensors="pt")
    encoding = {k: v.to(device) for k, v in encoding.items()}
    with torch.no_grad():
        outputs = dinov2_model(**encoding)
        logits_31 = outputs.logits.cpu().numpy()[0]

    # Softmax over 31 classes
    e = np.exp(logits_31 - np.max(logits_31))
    probs_31 = e / e.sum()

    # Map 31 probabilities → 6 by summing matched classes
    mapped_probs = np.zeros(NUM_MODEL_CLASSES, dtype=np.float64)
    for src_idx, dst_idx in DINOV2_TO_OUR_MAP.items():
        mapped_probs[dst_idx] += probs_31[src_idx]

    # Convert back to pseudo-logits (log scale) for compatibility with the rest of predict()
    # Add small epsilon to avoid log(0)
    mapped_probs = np.clip(mapped_probs, 1e-10, None)
    pseudo_logits = np.log(mapped_probs)
    return pseudo_logits


def predict(image_bytes: bytes, model_name: str = "vit"):
    start = time.time()

    # 1. Read image
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    original_size = img.size

    # 2. Run selected model
    try:
        if model_name == "dinov2":
            logits = _predict_dinov2(img)
            used_model_name = "DINOv2"
        elif model_name == "convnext":
            logits = _predict_timm(img, "convnext")
            used_model_name = "ConvNeXt V2"
        elif model_name == "efficientnet":
            logits = _predict_timm(img, "efficientnet")
            used_model_name = "EfficientNet-B1"
        else:
            logits = _predict_vit(img)
            used_model_name = "ViT"
            model_name = "vit"  # fallback

        # ─── บังคับตัด ACK ออก: set logit = -inf ก่อน softmax ──────
        for idx in EXCLUDED_CLASS_INDICES:
            logits[idx] = -np.inf  # ACK จะได้ probability = 0 หลัง softmax

        # Softmax
        e = np.exp(logits - np.max(logits))
        model_probs = e / e.sum()
    except Exception as ex:
        raise Exception(f"{used_model_name} prediction failed: {ex}")

    model_results = {used_model_name: {"probs": model_probs, "weight": 1.0}}
    
    # 3. Use model probabilities directly
    total_weight = sum(r["weight"] for r in model_results.values())
    ensemble_probs = model_probs.copy()

    # ─── Class Calibration: ปรับ weight ตามความสำคัญทางคลินิก ───────
    CLASS_CALIBRATION = np.array([
        0.0,   # ACK — ถูกตัดออก (logit = -inf แล้ว)
        1.3,   # BCC — Basal Cell Carcinoma (มะเร็ง, boost)
        1.4,   # MEL — Melanoma (ร้ายแรงสุด, boost มากหน่อย)
        0.6,   # NEV — Nevi (ไฝ, ลดลงเพราะข้อมูลเยอะ)
        1.3,   # SCC — Squamous Cell Carcinoma (มะเร็ง, boost)
        1.0,   # SEK — Seborrheic Keratosis (ปกติ)
    ])
    
    calibrated_probs = ensemble_probs * CLASS_CALIBRATION
    cal_total = calibrated_probs.sum()
    if cal_total > 0:
        calibrated_probs = calibrated_probs / cal_total
    
    elapsed = time.time() - start

    # 4. Map model classes → display classes (ข้าม ACK) ─────────────
    merged_probs = {dc: 0.0 for dc in DISPLAY_CLASSES}
    for i in range(NUM_MODEL_CLASSES):
        model_cls = MODEL_CLASSES[i]
        if model_cls not in CLASS_MERGE_MAP:
            continue  # ข้าม ACK
        display_cls = CLASS_MERGE_MAP[model_cls]
        merged_probs[display_cls] += float(calibrated_probs[i])

    # Re-normalize merged probs
    merged_total = sum(merged_probs.values())
    if merged_total > 0:
        merged_probs = {k: v / merged_total for k, v in merged_probs.items()}

    # Find top prediction from merged
    pred_class = max(merged_probs, key=merged_probs.get)
    info = CLASS_INFO[pred_class]
    confidence = merged_probs[pred_class]

    # Warnings
    warnings = []
    if confidence < 0.35:
        warnings.append(f"ความมั่นใจต่ำ ({confidence*100:.0f}%) — ควรปรึกษาแพทย์")

    results = []
    for dc in DISPLAY_CLASSES:
        ci = CLASS_INFO[dc]
        prob = merged_probs[dc]
        results.append({
            "class": dc,
            "name_th": ci["name_th"],
            "name_en": ci["name_en"],
            "probability": prob,
            "percentage": f"{prob*100:.1f}%",
            "color": ci["color"],
        })
    results.sort(key=lambda x: x["probability"], reverse=True)

    # Per-model breakdown for debug
    model_breakdown = {}
    for name, result in model_results.items():
        top_idx = int(np.argmax(result["probs"]))
        model_breakdown[name] = {
            "prediction": MODEL_CLASSES[top_idx],
            "confidence": f"{result['probs'][top_idx]*100:.1f}%",
            "weight": f"{result['weight']/total_weight*100:.0f}%",
        }

    # Model info for response
    model_info = MODEL_REGISTRY.get(model_name, MODEL_REGISTRY["vit"])

    return {
        "prediction": pred_class,
        "name_th": info["name_th"],
        "name_en": info["name_en"],
        "confidence": confidence,
        "confidence_pct": f"{confidence*100:.1f}%",
        "risk": info["risk"],
        "color": info["color"],
        "icon": info["icon"],
        "description": info["desc"],
        "advice": info["advice"],
        "recommendations": info.get("recommendations", []),
        "all_results": results,
        "elapsed_ms": f"{elapsed*1000:.0f}",
        "warnings": warnings,
        "is_uncertain": confidence < 0.35,
        "ensemble_models": len(model_results),
        "model_breakdown": model_breakdown,
        "model_used": {
            "key": model_name,
            "name": model_info["name"],
            "name_short": model_info["name_short"],
            "accuracy": model_info["accuracy"],
            "icon": model_info["icon"],
        },
        "debug": {
            "mode": "single",
            "active_models": list(model_results.keys()),
            "classes": NUM_DISPLAY_CLASSES,
            "top_class": pred_class,
            "image_size": f"{original_size[0]}x{original_size[1]}",
            "device": str(device),
            "model_key": model_name,
        }
    }

# ─── Skin Validation ────────────────────────────────────────
def validate_skin_image(img: Image.Image) -> dict:
    """ตรวจว่าภาพเป็นภาพถ่ายผิวหนังที่ชัดเจนหรือไม่ (เข้มงวด)"""
    w, h = img.size
    
    # 1. ขนาดเล็กเกินไป — ต้อง >= 100x100
    if w < 100 or h < 100:
        return {"valid": False, "reason": "ภาพเล็กเกินไป กรุณาใช้ภาพที่ใหญ่กว่า 100x100 px"}
    
    # 2. ตรวจสัดส่วนสีผิว (skin color detection)
    img_small = img.resize((128, 128))
    arr = np.array(img_small, dtype=np.float32)
    
    if len(arr.shape) < 3 or arr.shape[2] < 3:
        return {"valid": False, "reason": "กรุณาใช้ภาพสี (ไม่ใช่ขาวดำ)"}
    
    r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    
    # Skin color detection using multiple rules
    # Rule 1: RGB range for diverse skin tones
    skin_r = (r > 50) & (r < 255)
    skin_g = (g > 30) & (g < 230)
    skin_b = (b > 15) & (b < 210)
    skin_rg = (r > g) & (r > b)
    skin_mask1 = skin_r & skin_g & skin_b & skin_rg
    
    # Rule 2: YCbCr color space
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 128 - 0.169 * r - 0.331 * g + 0.500 * b
    cr = 128 + 0.500 * r - 0.419 * g - 0.081 * b
    skin_mask2 = (y > 60) & (cb > 77) & (cb < 127) & (cr > 133) & (cr < 173)
    
    skin_mask = skin_mask1 | skin_mask2
    skin_ratio = float(np.mean(skin_mask))
    
    # 3. ตรวจความชัด (Sharpness) — ใช้ Laplacian variance
    gray = np.mean(arr, axis=2)
    # Laplacian approximation
    from scipy.ndimage import laplace, sobel
    lap = laplace(gray)
    sharpness = float(np.var(lap))
    
    # 4. ตรวจ edge density — ภาพที่มีเส้นขอบเยอะ (ข้อความ, วัตถุ, screenshot) จะถูก reject
    edge_x = sobel(gray, axis=0)
    edge_y = sobel(gray, axis=1)
    edge_mag = np.sqrt(edge_x**2 + edge_y**2)
    edge_threshold = np.mean(edge_mag) + 1.5 * np.std(edge_mag)
    edge_ratio = float(np.mean(edge_mag > edge_threshold))
    
    # 5. ตรวจ entropy — ภาพ screenshot/ข้อความจะมี std ต่ำมาก
    std_val = float(np.std(gray))
    
    # 6. ตรวจว่ามีเนื้อหาเพียงพอ (ไม่ใช่ภาพขาว/ดำทั้งหมด)
    mean_brightness = float(np.mean(gray))
    
    # ─── ตัดสิน (เข้มงวด ≥80% skin + ห้าม edge เยอะ) ───
    
    # สีผิวน้อยเกินไป (ต้อง ≥80%)
    if skin_ratio < 0.80:
        return {
            "valid": False, 
            "reason": f"ภาพนี้มีสีผิวหนังเพียง {skin_ratio*100:.0f}% (ต้อง ≥80%) กรุณาถ่ายภาพผิวหนังให้ชัดเจนและเต็มเฟรม",
            "skin_ratio": f"{skin_ratio*100:.0f}%"
        }
    
    # มีเส้นขอบเยอะเกินไป (ข้อความ, วัตถุ, หลายสิ่ง)
    if edge_ratio > 0.25:
        return {
            "valid": False,
            "reason": f"ภาพมีรายละเอียด/เส้นขอบมากเกินไป ({edge_ratio*100:.0f}%) กรุณาถ่ายเฉพาะผิวหนัง ไม่ใช่ภาพที่มีข้อความ วัตถุ หรือฉากหลัง",
            "edge_ratio": f"{edge_ratio*100:.0f}%"
        }
    
    # ภาพไม่ชัด
    if sharpness < 30:
        return {
            "valid": False,
            "reason": f"ภาพไม่ชัดเพียงพอ กรุณาถ่ายภาพให้ชัดและอยู่นิ่ง",
            "sharpness": f"{sharpness:.1f}"
        }
    
    # ภาพสีเดียว
    if std_val < 12:
        return {
            "valid": False,
            "reason": "ภาพดูเหมือนสีเดียวกันทั้งหมด กรุณาใช้ภาพถ่ายจริงของผิวหนัง"
        }
    
    # สว่างหรือมืดเกินไป
    if mean_brightness < 30:
        return {
            "valid": False,
            "reason": "ภาพมืดเกินไป กรุณาถ่ายในที่มีแสงเพียงพอ"
        }
    if mean_brightness > 245:
        return {
            "valid": False,
            "reason": "ภาพสว่างเกินไป กรุณาลดความสว่างหรือหลีกเลี่ยงแสงจ้า"
        }
    
    return {
        "valid": True, 
        "skin_ratio": f"{skin_ratio*100:.0f}%",
        "edge_ratio": f"{edge_ratio*100:.0f}%",
        "sharpness": f"{sharpness:.1f}",
        "std": f"{std_val:.1f}"
    }


# ─── Image Preprocessing ─────────────────────────────────────
def remove_hair(arr: np.ndarray) -> np.ndarray:
    """
    ลบขน (Hair Removal) ด้วย Black-hat morphology + inpainting.
    
    ขั้นตอน:
    1. Grayscale → Black-hat transform (หาเส้นมืดบาง = ขน)
    2. Threshold หา hair mask
    3. Inpaint (แทนที่ pixel ขนด้วยสีรอบข้าง)
    """
    from scipy.ndimage import grey_closing, median_filter
    
    gray = np.mean(arr, axis=2).astype(np.uint8)
    
    # Black-hat: closing - original → เน้นเส้นมืดบาง (ขน)
    # ใช้ rectangular kernel ขนาดใหญ่
    kernel_size = max(15, min(arr.shape[0], arr.shape[1]) // 50)
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    closed = grey_closing(gray, size=(kernel_size, kernel_size))
    blackhat = closed.astype(np.int16) - gray.astype(np.int16)
    blackhat = np.clip(blackhat, 0, 255).astype(np.uint8)
    
    # Threshold: pixel ที่มีค่า blackhat สูง = ขน
    hair_threshold = 15
    hair_mask = blackhat > hair_threshold
    
    hair_ratio = float(np.mean(hair_mask))
    if hair_ratio < 0.001 or hair_ratio > 0.3:
        # ไม่มีขน หรือมากเกินไป (อาจผิดพลาด)
        return arr
    
    print(f"  🪒 Hair removal: {hair_ratio*100:.1f}% of pixels detected as hair")
    
    # Inpainting: แทนที่ pixel ขนด้วย median ของบริเวณรอบข้าง
    result = arr.copy()
    for c in range(3):  # R, G, B channels
        channel = arr[:, :, c].astype(np.float64)
        smoothed = median_filter(channel, size=7)
        result[:, :, c] = np.where(hair_mask, smoothed, channel).astype(np.uint8)
    
    return result


def normalize_illumination(arr: np.ndarray) -> np.ndarray:
    """
    ปรับแสงเงาให้สม่ำเสมอ (Illumination Normalization).
    
    ขั้นตอน:
    1. แยก luminance (ความสว่าง) จาก color
    2. Estimate illumination map ด้วย large Gaussian blur
    3. แก้ไข: pixel / illumination_map × target_brightness
    → ทำให้สีผิวทั้งภาพสม่ำเสมอ ไม่มีเงามืด/สว่างเกินไป
    """
    from scipy.ndimage import gaussian_filter
    
    result = arr.astype(np.float64)
    
    # แยกแต่ละ channel
    for c in range(3):
        channel = result[:, :, c]
        
        # Estimate illumination: large Gaussian blur
        sigma = max(arr.shape[0], arr.shape[1]) / 8
        illumination = gaussian_filter(channel, sigma=sigma)
        
        # หลีกเลี่ยง division by zero
        illumination = np.maximum(illumination, 1.0)
        
        # Target brightness = mean ของ illumination
        target = np.mean(illumination)
        
        # Correct: pixel × (target / local_illumination)
        corrected = channel * (target / illumination)
        result[:, :, c] = np.clip(corrected, 0, 255)
    
    print(f"  💡 Illumination normalized")
    return result.astype(np.uint8)


# ─── Skin Lesion UNet Segmentation ─────────────────────────
# Shifaa UNet — เทรนจาก HAM10000 สำหรับ skin lesion โดยเฉพาะ!
# Dice Score: 0.9175 | ขนาด: 7.78 MB | Input: 128×128 grayscale

import torch.nn as nn

class SkinLesionUNet(nn.Module):
    """UNet for skin lesion segmentation (Shifaa architecture)
    Architecture: Conv2d→ReLU→Conv2d→ReLU (no BatchNorm)
    """
    def __init__(self):
        super().__init__()
        # Encoder blocks: Conv→ReLU→Conv→ReLU
        self.enc1 = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(inplace=True))
        self.enc2 = nn.Sequential(nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(inplace=True))
        self.enc3 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True))
        self.enc4 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True))
        self.pool = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True))
        
        # Decoder
        self.up4 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec4 = nn.Sequential(nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True))
        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec3 = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True))
        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec2 = nn.Sequential(nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(inplace=True))
        self.up1 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dec1 = nn.Sequential(nn.Conv2d(32, 16, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(inplace=True))
        
        # Final
        self.final = nn.Conv2d(16, 1, 1)
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # Bottleneck
        b = self.bottleneck(self.pool(e4))
        
        # Decoder with skip connections
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        
        return torch.sigmoid(self.final(d1))

print("\n  🔬 Loading Skin Lesion UNet segmentation model...")
UNET_REPO = "Ahmed-Selem/Shifaa-Skin-Cancer-UNet-Segmentation"
try:
    from huggingface_hub import hf_hub_download
    
    # ดาวน์โหลด weights จาก HuggingFace
    weights_path = hf_hub_download(
        repo_id=UNET_REPO,
        filename="Shifaa-Skin-Cancer-UNet-Segmentation.pth"
    )
    
    seg_unet = SkinLesionUNet()
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    seg_unet.load_state_dict(state_dict)
    seg_unet.eval()
    seg_unet = seg_unet.to(device)
    unet_seg_available = True
    print(f"       ✅ Skin Lesion UNet loaded! (Dice: 0.9175)")
    print(f"       📦 Source: {UNET_REPO}")
except Exception as e:
    unet_seg_available = False
    print(f"       ⚠️ UNet not available: {e}")
    print(f"       → จะใช้ Otsu thresholding แทน")


def segment_lesion_unet(img: Image.Image) -> dict:
    """
    Segmentation ด้วย Shifaa UNet — เทรนจาก HAM10000 โดยเฉพาะ!
    Dice Score: 0.9175 | แม่นยำสำหรับ dermoscopy images
    
    ขั้นตอน:
    1. Resize เป็น 128×128 grayscale
    2. UNet ทำนาย binary mask
    3. Resize mask กลับเป็นขนาดเดิม
    4. ดึง bounding box จาก mask
    """
    arr = np.array(img)
    h, w = arr.shape[:2]
    
    # 1. Prepare input: resize to 128x128 grayscale, normalize to [0,1]
    gray = img.convert("L").resize((128, 128), Image.BILINEAR)
    gray_arr = np.array(gray).astype(np.float32) / 255.0
    input_tensor = torch.from_numpy(gray_arr).unsqueeze(0).unsqueeze(0)  # (1,1,128,128)
    input_tensor = input_tensor.to(device)
    
    # 2. Run UNet
    with torch.no_grad():
        pred = seg_unet(input_tensor)  # (1,1,128,128)
    
    # 3. Post-process: threshold + resize back
    mask_small = pred.squeeze().cpu().numpy()  # (128,128) float
    mask_binary = (mask_small > 0.6).astype(np.uint8)  # Threshold สูงขึ้น (0.6) เพื่อลด false positive
    
    # Resize mask back to original size
    mask_full = np.array(
        Image.fromarray(mask_binary * 255).resize((w, h), Image.NEAREST)
    ) > 127
    
    mask_ratio = float(np.mean(mask_full))
    
    # กรอง: mask ต้องมีขนาดเหมาะสม
    if mask_ratio > 0.85 or mask_ratio < 0.005:
        return {"found": False, "bbox": None, "mask_ratio": mask_ratio}
    
    # Confidence check: ค่าเฉลี่ย prediction ภายใน mask ต้อง > 0.70
    confidence = float(np.mean(mask_small[mask_binary > 0])) if np.any(mask_binary) else 0.0
    if confidence < 0.70:
        print(f"  ⚠️ Low UNet confidence: {confidence:.2f} — likely normal skin")
        return {"found": False, "bbox": None, "mask_ratio": mask_ratio}
    
    # ─── Color Contrast Check: เปรียบเทียบสีใน vs นอก mask ───
    # ถ้าสีใน/นอก mask ใกล้กัน = ไม่มีรอยโรค (ผิวปกติ)
    gray_full = np.array(img.convert("L"))
    inside_pixels = gray_full[mask_full]
    outside_pixels = gray_full[~mask_full]
    
    if len(inside_pixels) > 0 and len(outside_pixels) > 0:
        mean_inside = float(np.mean(inside_pixels))
        mean_outside = float(np.mean(outside_pixels))
        contrast_diff = abs(mean_inside - mean_outside)
        
        if contrast_diff < 15:
            print(f"  ⚠️ Low contrast: inside={mean_inside:.0f} outside={mean_outside:.0f} diff={contrast_diff:.1f} — likely normal skin")
            return {"found": False, "bbox": None, "mask_ratio": mask_ratio}
    
    # 4. Get bounding box from mask
    rows = np.any(mask_full, axis=1)
    cols = np.any(mask_full, axis=0)
    if not np.any(rows) or not np.any(cols):
        return {"found": False, "bbox": None, "mask_ratio": mask_ratio}
    
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    
    # Add padding (10%)
    pad_x = int((x_max - x_min) * 0.10)
    pad_y = int((y_max - y_min) * 0.10)
    x_min = max(0, x_min - pad_x)
    y_min = max(0, y_min - pad_y)
    x_max = min(w, x_max + pad_x)
    y_max = min(h, y_max + pad_y)
    
    print(f"  ✅ UNet confirmed lesion: conf={confidence:.2f}, contrast_diff={contrast_diff:.1f}")
    
    return {
        "found": True,
        "bbox": (int(x_min), int(y_min), int(x_max), int(y_max)),
        "mask_ratio": mask_ratio,
        "mask": mask_full,
        "method": "UNet (Skin Lesion)",
        "iou_score": confidence,
    }


def segment_lesion(img: Image.Image) -> dict:
    """
    Lesion segmentation ด้วย UNet เท่านั้น
    ถ้าไม่เจอรอยโรค → return found=False (ผิวหนังปกติ)
    """
    if unet_seg_available:
        try:
            result = segment_lesion_unet(img)
            if result["found"]:
                print(f"  🔬 UNet segmented: ({result['bbox'][0]},{result['bbox'][1]})-({result['bbox'][2]},{result['bbox'][3]}) conf={result.get('iou_score', 0):.2f}")
            else:
                print(f"  ✅ UNet: ไม่พบรอยโรค → ผิวหนังปกติ")
            return result
        except Exception as e:
            print(f"  ⚠️ UNet error: {e}")
    
    # UNet ไม่พร้อม → ถือว่าไม่พบรอยโรค
    print(f"  ⚠️ UNet not available, treating as normal skin")
    return {"found": False, "bbox": None, "mask_ratio": 0.0}


# ─── FastAPI ───────────────────────────────────────────────
app = FastAPI(title="Skin Cancer Scanner", version="5.0")

templates_dir = BASE_DIR / "templates"
templates_dir.mkdir(exist_ok=True)
templates = Jinja2Templates(directory=str(templates_dir))

static_dir = BASE_DIR / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# ─── Real-time Camera Frame Validation ─────────────────────
def validate_camera_frame(img: Image.Image) -> dict:
    """
    ตรวจเฟรมกล้องว่าเป็นภาพผิวหนัง close-up จริงหรือไม่
    ใช้ 5 เทคนิค Computer Vision:
    1. Skin Color (YCbCr) — ผิวหนังต้อง >70%
    2. Spatial Uniformity — แบ่ง grid ตรวจกระจายสม่ำเสมอ  
    3. Edge Density — ใบหน้ามี edge เยอะ ผิว close-up มีน้อย
    4. Dark Cluster — ตา/ผม/คิ้ว = กลุ่ม pixel มืด
    5. Sharpness — ภาพต้อง focus ชัด
    """
    from scipy.ndimage import uniform_filter, laplace
    
    # Resize เล็กเพื่อความเร็ว
    img_small = img.resize((128, 128))
    arr = np.array(img_small, dtype=np.float32)
    r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    gray = (0.299 * r + 0.587 * g + 0.114 * b)
    
    checks = {}
    
    # ── Check 1: Skin Color Ratio (YCbCr) ──
    y  = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 128 - 0.169 * r - 0.331 * g + 0.500 * b
    cr = 128 + 0.500 * r - 0.419 * g - 0.081 * b
    skin_mask = (y > 60) & (cb > 77) & (cb < 127) & (cr > 133) & (cr < 173)
    skin_ratio = float(np.mean(skin_mask))
    checks["skin_ratio"] = round(skin_ratio, 3)
    checks["skin_ok"] = skin_ratio > 0.40
    
    # ── Check 2: Spatial Uniformity (Grid Variance) ──
    # แบ่งภาพเป็น 4x4 grid ตรวจว่า brightness สม่ำเสมอ
    # ผิว close-up: ทุก cell มี mean brightness ใกล้กัน
    # ใบหน้า: cell ตา/ผม มืด, cell หน้าผาก สว่าง
    grid_size = 4
    cell_h, cell_w = 128 // grid_size, 128 // grid_size
    cell_means = []
    cell_skin_ratios = []
    for gy in range(grid_size):
        for gx in range(grid_size):
            cell = gray[gy*cell_h:(gy+1)*cell_h, gx*cell_w:(gx+1)*cell_w]
            cell_means.append(float(np.mean(cell)))
            cell_skin = skin_mask[gy*cell_h:(gy+1)*cell_h, gx*cell_w:(gx+1)*cell_w]
            cell_skin_ratios.append(float(np.mean(cell_skin)))
    
    grid_std = float(np.std(cell_means))
    # ผิวซูมชัด: std < 18, ใบหน้า/วัตถุ: std > 18
    checks["grid_std"] = round(grid_std, 2)
    checks["uniform_ok"] = grid_std < 18
    
    # ตรวจว่า skin กระจายทั่วทุก cell (ไม่ใช่แค่บาง cell)
    low_skin_cells = sum(1 for s in cell_skin_ratios if s < 0.3)
    checks["low_skin_cells"] = low_skin_cells
    checks["spread_ok"] = low_skin_cells <= 2  # อนุญาตไม่เกิน 2 cell ที่ skin น้อย
    
    # ── Check 3: Edge Density (Laplacian) ──
    # ใบหน้ามี edge เยอะ (ตา จมูก ปาก) ผิว close-up มีน้อย
    lap = laplace(gray.astype(np.float64))
    edge_density = float(np.mean(np.abs(lap)))
    checks["edge_density"] = round(edge_density, 2)
    checks["edge_ok"] = edge_density < 5.0  # ผิว close-up < 5, ใบหน้า > 5
    
    # ── Check 4: Dark Cluster Detection ──
    # ตรวจ pixel มืดมาก (ตา, ผม, คิ้ว)
    dark_mask = gray < 50
    dark_ratio = float(np.mean(dark_mask))
    checks["dark_ratio"] = round(dark_ratio, 3)
    checks["dark_ok"] = dark_ratio < 0.05  # ผิว close-up แทบไม่มี pixel มืด
    
    # ── Check 5: Sharpness (Focus Quality) ──
    lap_var = float(np.var(lap))
    checks["sharpness"] = round(lap_var, 2)
    checks["sharp_ok"] = lap_var > 3.0  # ต้อง focus ชัด
    
    # ── Final Decision ──
    # เช็คแค่ skin + sharpness (ไม่ต้องเช็คใบหน้า)
    is_valid = checks["skin_ok"] and checks.get("sharp_ok", True)
    
    # สร้าง hint
    if is_valid:
        hint = "พร้อมถ่ายภาพ"
        hint_detail = "กดปุ่มเพื่อถ่ายภาพ"
    elif not checks["skin_ok"]:
        hint = "ซูมให้ชิดผิวหนัง"
        hint_detail = f"ต้องเห็นผิวหนัง >40% (ตอนนี้ {skin_ratio*100:.0f}%)"
    elif not checks.get("sharp_ok", True):
        hint = "ภาพไม่ชัด"
        hint_detail = "กรุณาถ่ายให้ชัดและอยู่นิ่ง"
    else:
        hint = "ปรับตำแหน่ง"
        hint_detail = "วางผิวหนังให้อยู่ในกรอบ"
    
    return {
        "valid": is_valid,
        "hint": hint,
        "hint_detail": hint_detail,
        "checks": checks,
    }


@app.post("/api/validate-frame")
async def api_validate_frame(file: UploadFile = File(...)):
    """Real-time camera frame validation — ตรวจว่าเป็นผิวหนัง close-up"""
    try:
        contents = await file.read()
        if len(contents) == 0:
            return JSONResponse({"valid": False, "hint": "ไม่มีภาพ"})
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        result = validate_camera_frame(img)
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"valid": False, "hint": f"Error: {str(e)}"})


@app.post("/api/validate-upload")
async def api_validate_upload(file: UploadFile = File(...)):
    """ตรวจภาพที่อัปโหลดว่าเป็นผิวหนังจริงหรือไม่ (เข้มงวด ≥80%)"""
    try:
        contents = await file.read()
        if len(contents) == 0:
            return JSONResponse({"valid": False, "reason": "ไม่มีภาพ", "skin_ratio": 0})
        
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        w, h = img.size
        
        # ขนาดขั้นต่ำ
        if w < 100 or h < 100:
            return JSONResponse({"valid": False, "reason": "ภาพเล็กเกินไป กรุณาใช้ภาพที่ใหญ่กว่า 100x100 px", "skin_ratio": 0})
        
        # Resize เพื่อประมวลผลเร็ว
        img_small = img.resize((128, 128))
        arr = np.array(img_small, dtype=np.float32)
        
        if len(arr.shape) < 3 or arr.shape[2] < 3:
            return JSONResponse({"valid": False, "reason": "กรุณาใช้ภาพสี (ไม่ใช่ขาวดำ)", "skin_ratio": 0})
        
        r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
        
        # Skin detection (RGB + YCbCr)
        skin_r = (r > 50) & (r < 255)
        skin_g = (g > 30) & (g < 230)
        skin_b = (b > 15) & (b < 210)
        skin_rg = (r > g) & (r > b)
        skin_mask1 = skin_r & skin_g & skin_b & skin_rg
        
        y = 0.299 * r + 0.587 * g + 0.114 * b
        cb = 128 - 0.169 * r - 0.331 * g + 0.500 * b
        cr = 128 + 0.500 * r - 0.419 * g - 0.081 * b
        skin_mask2 = (y > 60) & (cb > 77) & (cb < 127) & (cr > 133) & (cr < 173)
        
        skin_mask = skin_mask1 | skin_mask2
        skin_ratio = float(np.mean(skin_mask))
        skin_pct = round(skin_ratio * 100)
        
        # Sharpness - Laplacian
        gray = np.mean(arr, axis=2)
        from scipy.ndimage import laplace
        lap = laplace(gray)
        sharpness = float(np.var(lap))
        
        # Brightness
        mean_brightness = float(np.mean(gray))
        std_val = float(np.std(gray))
        
        # ─── ตัดสิน (ต้อง ≥80% skin) ───
        if skin_pct < 80:
            return JSONResponse({
                "valid": False,
                "reason": f"ภาพนี้มีสีผิวหนังเพียง {skin_pct}% (ต้อง ≥80%) กรุณาใช้ภาพถ่ายผิวหนังโดยตรง",
                "skin_ratio": skin_pct
            })
        
        if sharpness < 30:
            return JSONResponse({
                "valid": False,
                "reason": "ภาพไม่ชัดเพียงพอ กรุณาถ่ายภาพให้ชัดและนิ่ง",
                "skin_ratio": skin_pct
            })
        
        if mean_brightness < 30:
            return JSONResponse({
                "valid": False,
                "reason": "ภาพมืดเกินไป กรุณาถ่ายในที่มีแสงเพียงพอ",
                "skin_ratio": skin_pct
            })
        
        if mean_brightness > 245:
            return JSONResponse({
                "valid": False,
                "reason": "ภาพสว่างเกินไป กรุณาลดแสง",
                "skin_ratio": skin_pct
            })
        
        if std_val < 12:
            return JSONResponse({
                "valid": False,
                "reason": "ภาพดูเหมือนสีเดียว กรุณาใช้ภาพถ่ายจริง",
                "skin_ratio": skin_pct
            })
        
        return JSONResponse({
            "valid": True,
            "skin_ratio": skin_pct,
            "sharpness": round(sharpness, 1),
            "reason": f"ผ่าน! ผิวหนัง {skin_pct}%"
        })
    except Exception as e:
        return JSONResponse({"valid": False, "reason": f"เกิดข้อผิดพลาด: {str(e)}", "skin_ratio": 0})


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    try:
        return templates.TemplateResponse(name="index.html", request=request)
    except TypeError:
        return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/models")
async def api_models():
    """List all available models and their status"""
    models = []
    for key, info in MODEL_REGISTRY.items():
        models.append({
            "key": key,
            "name": info["name"],
            "name_short": info["name_short"],
            "description": info["description"],
            "accuracy": info["accuracy"],
            "active": info["active"],
            "icon": info["icon"],
        })
    return JSONResponse({"models": models, "default": "vit"})


@app.post("/api/predict")
async def api_predict(file: UploadFile = File(...), model_name: str = Form("vit")):
    try:
        import base64
        
        contents = await file.read()
        if len(contents) == 0:
            return JSONResponse({"error": "Empty file"}, status_code=400)
        
        # Open image
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        original_size = img.size
        
        # 1. Validate: เป็นภาพผิวหนังหรือไม่?
        validation = validate_skin_image(img)
        if not validation["valid"]:
            return JSONResponse({
                "error": "ภาพไม่เหมาะสม",
                "reason": validation["reason"],
                "is_rejected": True,
                "icon": "🚫",
                "advice": "กรุณาถ่ายภาพผิวหนังโดยตรง ให้เห็นผิวหนังชัดเจน",
            })
        
        # 2. Auto Lesion Segmentation
        seg_result = segment_lesion(img)
        seg_image_b64 = None
        seg_bbox_str = None
        
        # ─── ถ้าไม่เจอรอยโรค → ตอบ "ผิวหนังปกติ" ทันที (ไม่ต้อง classify) ───
        if not seg_result["found"] or not seg_result.get("bbox"):
            print(f"  ✅ No lesion found → ผิวหนังปกติ")
            return JSONResponse({
                "no_lesion_found": True,
                "prediction": "ผิวหนังปกติ (Normal Skin)",
                "confidence": 1.0,
                "risk": "none",
                "icon": "✅",
                "seg_image": None,
                "seg_bbox": None,
                "segmented": False,
                "skin_check": validation,
            })
        
        # ─── เจอรอยโรค → วาดขอบ + crop + classify ───
        x1, y1, x2, y2 = seg_result["bbox"]
        lesion_mask = seg_result.get("mask")
        
        # วาดขอบสวยๆ — เบลอนอก, ชัดใน, เส้นเรียบ
        vis_arr = np.array(img).copy()
        if lesion_mask is not None:
            from scipy.ndimage import binary_dilation, gaussian_filter
            from PIL import ImageFilter
            
            # 1. สร้าง smooth mask (sigma สูง = เส้นเรียบ)
            mask_float = lesion_mask.astype(np.float64)
            smooth_mask = gaussian_filter(mask_float, sigma=5.0)  # sigma สูงขึ้น = เรียบมาก
            inside_mask = smooth_mask > 0.5
            
            # 2. เบลอพื้นหลัง (นอก mask) + ทำให้มืด
            blurred_img = Image.fromarray(vis_arr).filter(ImageFilter.GaussianBlur(radius=15))
            blurred_arr = np.array(blurred_img)
            frosted = (blurred_arr.astype(np.float64) * 0.45).astype(np.uint8)
            
            # 3. รวม: ข้างในชัด + ข้างนอกเบลอขุ่น
            for c in range(3):
                vis_arr[:, :, c] = np.where(
                    inside_mask,
                    vis_arr[:, :, c],  # ข้างในไม่แตะ
                    frosted[:, :, c]  # ข้างนอกเบลอขุ่น
                )
            
            # 4. เส้น Contour สวย — gradient glow
            contour_band = (smooth_mask > 0.3) & (smooth_mask < 0.7)
            thick_contour = binary_dilation(contour_band, iterations=1)
            
            # Glow ชั้นที่ 1 — สีฟ้าเข้ม อ่อนๆ (กว้าง)
            glow_wide = binary_dilation(thick_contour, iterations=4)
            glow_wide_only = glow_wide & ~thick_contour
            vis_arr[glow_wide_only] = np.clip(
                vis_arr[glow_wide_only].astype(np.float64) * 0.7 + np.array([0, 50, 80]) * 0.3,
                0, 255
            ).astype(np.uint8)
            
            # Glow ชั้นที่ 2 — สีฟ้าสว่าง (แคบ)
            glow_narrow = binary_dilation(thick_contour, iterations=2)
            glow_narrow_only = glow_narrow & ~thick_contour
            vis_arr[glow_narrow_only] = np.clip(
                vis_arr[glow_narrow_only].astype(np.float64) * 0.5 + np.array([0, 140, 200]) * 0.5,
                0, 255
            ).astype(np.uint8)
            
            # เส้น Contour หลัก — สีฟ้าสดใส
            vis_arr[thick_contour] = np.array([0, 210, 255])
        
        vis_img = Image.fromarray(vis_arr)
        buf_vis = io.BytesIO()
        vis_img.save(buf_vis, format="JPEG", quality=90)
        seg_image_b64 = base64.b64encode(buf_vis.getvalue()).decode()
        
        # Crop เฉพาะรอยโรค
        img = img.crop((x1, y1, x2, y2))
        crop_w, crop_h = x2 - x1, y2 - y1
        seg_method = seg_result.get("method", "Auto")
        iou_str = f" (conf: {seg_result.get('iou_score', 0):.0%})" if seg_result.get("iou_score") else ""
        seg_bbox_str = f"{crop_w}x{crop_h}px — {seg_method}{iou_str} (จากต้นฉบับ {original_size[0]}x{original_size[1]})"
        print(f"  ✂️ Lesion cropped: ({x1},{y1})-({x2},{y2}) = {crop_w}x{crop_h}")
        
        # Preprocessing
        img_arr = np.array(img)
        img_arr = remove_hair(img_arr)
        img_arr = normalize_illumination(img_arr)
        img = Image.fromarray(img_arr)
        
        # สร้างภาพ preprocessed
        buf_pp = io.BytesIO()
        img.save(buf_pp, format="JPEG", quality=90)
        preprocessed_b64 = base64.b64encode(buf_pp.getvalue()).decode()
        
        # Predict with selected model
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=95)
        result = predict(buf.getvalue(), model_name=model_name)
        
        result["seg_image"] = seg_image_b64
        result["seg_bbox"] = seg_bbox_str
        result["segmented"] = True
        result["skin_check"] = validation
        result["preprocessed_image"] = preprocessed_b64
        result["no_lesion_found"] = False
        
        return JSONResponse(result)
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "version": "9.0-multi",
        "mode": "multi-model",
        "models": {k: {"active": v["active"], "name": v["name"]} for k, v in MODEL_REGISTRY.items()},
        "active_models": sum(1 for m in MODEL_REGISTRY.values() if m["active"]),
        "classes": NUM_DISPLAY_CLASSES,
        "device": str(device),
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

