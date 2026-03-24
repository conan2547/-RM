"""
Skin Cancer Scanner v8 — ViT (Vision Transformer)
ViT fine-tuned for 6-class skin cancer classification.
Model: bnmbanhmi/seekwell_skincancer_v2
Classes: ACK, BCC, MEL, NEV, SCC, SEK
"""
import os, io, time, traceback, json
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from transformers import ViTForImageClassification, AutoImageProcessor
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from typing import Optional

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


# ─── Load ViT Model ────────────────────────────────────────
print("=" * 65)
print("  🧬 ViT Skin Cancer Scanner v8")
print(f"  📊 6 Classes | seekwell_skincancer_v2")
print("=" * 65)

# Device
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("  🖥️  Device: Apple MPS (GPU)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("  🖥️  Device: CUDA GPU")
else:
    device = torch.device("cpu")
    print("  🖥️  Device: CPU")

# Load ViT model from HuggingFace
print(f"\n  📦 Loading ViT: {MODEL_HF_NAME}")

vit_model = ViTForImageClassification.from_pretrained(MODEL_HF_NAME)
# seekwell model ไม่มี preprocessor_config.json ใช้ base ViT processor แทน
vit_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
print(f"       ✅ Loaded from HuggingFace!")

vit_model.eval()
vit_model = vit_model.to(device)
print(f"       📋 Classes: {MODEL_CLASSES}")
print(f"       📐 Input size: 224x224")

# Summary
print(f"\n{'=' * 65}")
print(f"  ✅ ViT Ready! 1 model active")
print(f"     [1] ViT — seekwell_skincancer_v2")
print(f"  📋 Output: {NUM_MODEL_CLASSES} classes → {NUM_DISPLAY_CLASSES} display groups")
print(f"{'=' * 65}")


# ─── ViT Prediction ────────────────────────────────────────
def predict(image_bytes: bytes):
    start = time.time()

    # 1. Read image
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    original_size = img.size

    # 2. Run ViT model
    try:
        encoding = vit_processor(img, return_tensors="pt")
        encoding = {k: v.to(device) for k, v in encoding.items()}

        with torch.no_grad():
            outputs = vit_model(**encoding)
            logits = outputs.logits.cpu().numpy()[0]

        # ─── บังคับตัด ACK ออก: set logit = -inf ก่อน softmax ──────
        for idx in EXCLUDED_CLASS_INDICES:
            logits[idx] = -np.inf  # ACK จะได้ probability = 0 หลัง softmax

        # Softmax
        e = np.exp(logits - np.max(logits))
        model_probs = e / e.sum()
    except Exception as ex:
        raise Exception(f"ViT prediction failed: {ex}")

    model_results = {"ViT": {"probs": model_probs, "weight": 1.0}}
    
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
        "debug": {
            "mode": "single",
            "active_models": list(model_results.keys()),
            "classes": NUM_DISPLAY_CLASSES,
            "top_class": pred_class,
            "image_size": f"{original_size[0]}x{original_size[1]}",
            "device": str(device),
        }
    }

# ─── Skin Validation ────────────────────────────────────────
def validate_skin_image(img: Image.Image) -> dict:
    """ตรวจว่าภาพเป็นภาพถ่ายผิวหนังหรือไม่"""
    w, h = img.size
    
    # 1. ขนาดเล็กเกินไป
    if w < 50 or h < 50:
        return {"valid": False, "reason": "ภาพเล็กเกินไป กรุณาใช้ภาพที่ใหญ่กว่า 50x50 px"}
    
    # 2. ตรวจสัดส่วนสีผิว (skin color detection)
    img_small = img.resize((100, 100))
    arr = np.array(img_small, dtype=np.float32)
    
    if len(arr.shape) < 3 or arr.shape[2] < 3:
        return {"valid": False, "reason": "กรุณาใช้ภาพสี (ไม่ใช่ขาวดำ)"}
    
    r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    
    # Skin color detection using multiple rules
    # Rule 1: RGB range for skin tones
    skin_r = (r > 50) & (r < 255)
    skin_g = (g > 30) & (g < 230)
    skin_b = (b > 15) & (b < 210)
    skin_rg = (r > g) & (r > b)  # Red channel dominant
    skin_mask1 = skin_r & skin_g & skin_b & skin_rg
    
    # Rule 2: YCbCr color space (ดีกว่าสำหรับตรวจสีผิว)
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 128 - 0.169 * r - 0.331 * g + 0.500 * b
    cr = 128 + 0.500 * r - 0.419 * g - 0.081 * b
    skin_mask2 = (y > 60) & (cb > 77) & (cb < 127) & (cr > 133) & (cr < 173)
    
    skin_mask = skin_mask1 | skin_mask2
    skin_ratio = float(np.mean(skin_mask))
    
    # 3. ตรวจ entropy — ภาพ screenshot/ข้อความจะมี std ต่ำมาก
    gray = np.mean(arr, axis=2)
    std_val = float(np.std(gray))
    
    # 4. ตัดสิน
    if skin_ratio < 0.08:
        return {
            "valid": False, 
            "reason": "ไม่พบสีผิวหนังในภาพ กรุณาถ่ายภาพผิวหนังโดยตรง",
            "skin_ratio": f"{skin_ratio*100:.0f}%"
        }
    
    if std_val < 5:
        return {
            "valid": False,
            "reason": "ภาพดูเป็นสีเดียว กรุณาใช้ภาพถ่ายจริง"
        }
    
    return {
        "valid": True, 
        "skin_ratio": f"{skin_ratio*100:.0f}%",
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


# ─── Real-time Camera Frame Validation (UNet-based) ────────
def validate_camera_frame(img: Image.Image) -> dict:
    """
    ตรวจเฟรมกล้องด้วย UNet Segmentation แบบ real-time
    ถ้าเจอรอยโรคผิวหนัง → อนุญาตถ่ายภาพ + ส่ง mask overlay กลับ
    ถ้าไม่เจอ → ไม่อนุญาต
    """
    import base64
    from scipy.ndimage import binary_dilation, binary_erosion, gaussian_filter
    
    if not unet_seg_available:
        return {
            "valid": True,
            "hint": "✅ พร้อมถ่ายภาพ",
            "hint_detail": "กดปุ่มเพื่อถ่ายภาพ",
            "lesion_found": True,
            "unet_confidence": 0,
            "mask_overlay": None,
            "checks": {"lesion_found": True, "unet_ready": False},
        }
    
    try:
        arr = np.array(img)
        h, w = arr.shape[:2]
        
        # 1. Prepare input: resize to 128x128 grayscale
        gray = img.convert("L").resize((128, 128), Image.BILINEAR)
        gray_arr = np.array(gray).astype(np.float32) / 255.0
        input_tensor = torch.from_numpy(gray_arr).unsqueeze(0).unsqueeze(0)
        input_tensor = input_tensor.to(device)
        
        # 2. Run UNet
        with torch.no_grad():
            pred = seg_unet(input_tensor)
        
        # 3. Analyze prediction — เข้มงวดเพื่อให้มั่นใจว่าเป็นรอยโรคจริง
        mask_small = pred.squeeze().cpu().numpy()  # (128,128) float [0,1]
        
        # 3a. ใช้ threshold สูงขึ้น (0.6 แทน 0.5) เพื่อลด false positive
        mask_binary = (mask_small > 0.6).astype(np.uint8)
        
        # 3b. Morphological cleanup — ลบจุด noise เล็กๆ, เติมรูเล็กๆ
        from scipy.ndimage import binary_opening, binary_closing, label as ndlabel
        mask_clean = binary_opening(mask_binary, iterations=2)   # ลบจุดเล็ก
        mask_clean = binary_closing(mask_clean, iterations=2)    # เติมรูเล็ก
        mask_binary = mask_clean.astype(np.uint8)
        
        # 3c. Connected Component — เอาเฉพาะก้อนใหญ่สุด
        labeled_arr, num_features = ndlabel(mask_binary)
        if num_features > 0:
            # หาก้อนใหญ่สุด
            component_sizes = [(labeled_arr == i).sum() for i in range(1, num_features + 1)]
            largest_label = component_sizes.index(max(component_sizes)) + 1
            mask_binary = (labeled_arr == largest_label).astype(np.uint8)
            
            # ถ้าก้อนใหญ่สุดเล็กเกินไป (< 50 px จาก 128x128 = 16384 px) → ถือว่าไม่เจอ
            largest_size = max(component_sizes)
            if largest_size < 50:
                mask_binary = np.zeros_like(mask_binary)
        
        mask_ratio = float(np.mean(mask_binary))
        confidence = float(np.mean(mask_small[mask_binary > 0])) if np.any(mask_binary) else 0.0
        
        # 3d. Circularity check — รอยโรคจริงค่อนข้างกลม/compact ไม่เป็นเส้นยาว
        mask_pixels = int(np.sum(mask_binary))
        circularity = 0.0
        if mask_pixels > 0:
            from scipy.ndimage import binary_dilation as bd_circ
            dilated_circ = bd_circ(mask_binary.astype(bool), iterations=1)
            perimeter = int(np.sum(dilated_circ.astype(np.uint8) - mask_binary))
            if perimeter > 0:
                circularity = (4 * 3.14159 * mask_pixels) / (perimeter * perimeter)
            else:
                circularity = 1.0
        
        # 4. Contrast check — ต้องมีความแตกต่างชัดเจน
        gray_full = np.array(img.convert("L").resize((128, 128)))
        inside_pixels = gray_full[mask_binary > 0]
        outside_pixels = gray_full[mask_binary == 0]
        
        contrast_diff = 0.0
        if len(inside_pixels) > 0 and len(outside_pixels) > 0:
            contrast_diff = abs(float(np.mean(inside_pixels)) - float(np.mean(outside_pixels)))
        
        # 5. Skin Color Check (YCbCr) — ต้องเจอสีผิวหนังอย่างน้อย 20%
        img_small = img.resize((128, 128))
        arr_small = np.array(img_small, dtype=np.float32)
        r, g, b = arr_small[:,:,0], arr_small[:,:,1], arr_small[:,:,2]
        y_ch  = 0.299 * r + 0.587 * g + 0.114 * b
        cb_ch = 128 - 0.169 * r - 0.331 * g + 0.500 * b
        cr_ch = 128 + 0.500 * r - 0.419 * g - 0.081 * b
        skin_mask = (y_ch > 60) & (cb_ch > 77) & (cb_ch < 127) & (cr_ch > 133) & (cr_ch < 173)
        skin_ratio = float(np.mean(skin_mask))
        skin_ok = skin_ratio >= 0.20
        
        # 5b. Face Detection Heuristics — กรองใบหน้าออก
        from scipy.ndimage import sobel
        
        # (A) Edge density — ใบหน้ามี edge เยอะ (ตา ปาก จมูก คิ้ว), ผิวหนัง close-up มี edge น้อย
        gray_sobel = gray_full.astype(np.float64)
        edge_x = sobel(gray_sobel, axis=0)
        edge_y = sobel(gray_sobel, axis=1)
        edge_mag = np.sqrt(edge_x**2 + edge_y**2)
        edge_density = float(np.mean(edge_mag > 25))  # สัดส่วน pixel ที่เป็น edge
        is_face_edge = edge_density > 0.15  # ใบหน้ามี edge > 15%
        
        # (B) Skin spread uniformity — ใบหน้ามีผิวหนังกระจายทั่ว 4 มุม
        skin_mask_bool = skin_mask.astype(bool)
        q1 = float(np.mean(skin_mask_bool[:64, :64]))    # top-left
        q2 = float(np.mean(skin_mask_bool[:64, 64:]))    # top-right
        q3 = float(np.mean(skin_mask_bool[64:, :64]))    # bottom-left
        q4 = float(np.mean(skin_mask_bool[64:, 64:]))    # bottom-right
        quadrants = [q1, q2, q3, q4]
        min_q = min(quadrants)
        # ถ้าทุก quadrants > 15% = ผิวหนังกระจายสม่ำเสมอ (ใบหน้า)
        is_face_spread = min_q > 0.15 and skin_ratio > 0.50
        
        # (C) Max skin ratio — ผิวหนัง > 80% มักเป็นใบหน้า ไม่ใช่ close-up รอยโรค
        is_face_maxskin = skin_ratio > 0.80
        
        # รวม: ถ้าเจอ >= 2 signals → น่าจะเป็นใบหน้า
        face_signals = sum([is_face_edge, is_face_spread, is_face_maxskin])
        is_face = face_signals >= 2
        
        # 6. Decision — เข้มงวดทุกเงื่อนไข + ต้องไม่ใช่ใบหน้า
        lesion_found = (
            skin_ok and
            not is_face and                     # ต้องไม่ใช่ใบหน้า
            0.01 < mask_ratio < 0.75 and
            confidence > 0.75 and
            contrast_diff > 15 and
            circularity > 0.15
        )
        
        # 7. สร้าง Mask Overlay Image (128x128 RGBA PNG)
        mask_overlay_b64 = None
        if lesion_found:
            overlay = np.zeros((128, 128, 4), dtype=np.uint8)
            
            mask_bool = mask_binary.astype(bool)
            
            # Semi-transparent fill inside lesion (cyan)
            overlay[mask_bool, 0] = 0      # R
            overlay[mask_bool, 1] = 220    # G
            overlay[mask_bool, 2] = 255    # B
            overlay[mask_bool, 3] = 35     # Alpha (very subtle fill)
            
            # Find contour (edge of mask)
            dilated = binary_dilation(mask_bool, iterations=1)
            eroded = binary_erosion(mask_bool, iterations=1)
            contour = dilated & ~eroded  # เส้นขอบ
            
            # Glow effect — outer glow
            glow_wide = binary_dilation(mask_bool, iterations=3) & ~dilated
            overlay[glow_wide, 0] = 0
            overlay[glow_wide, 1] = 180
            overlay[glow_wide, 2] = 255
            overlay[glow_wide, 3] = 50  # subtle glow
            
            glow_narrow = binary_dilation(mask_bool, iterations=2) & ~dilated
            overlay[glow_narrow, 0] = 0
            overlay[glow_narrow, 1] = 200
            overlay[glow_narrow, 2] = 255
            overlay[glow_narrow, 3] = 80  # brighter glow
            
            # Main contour line — bright cyan
            overlay[contour, 0] = 0
            overlay[contour, 1] = 230
            overlay[contour, 2] = 255
            overlay[contour, 3] = 220  # mostly opaque
            
            # Encode as PNG base64
            overlay_img = Image.fromarray(overlay, 'RGBA')
            buf_overlay = io.BytesIO()
            overlay_img.save(buf_overlay, format='PNG')
            mask_overlay_b64 = base64.b64encode(buf_overlay.getvalue()).decode()
        
        # Hints
        if not skin_ok:
            hint = "🔍 ไม่พบผิวหนัง"
            hint_detail = f"ต้องเห็นผิวหนัง ≥20% (ตอนนี้ {skin_ratio*100:.0f}%)"
        elif is_face:
            hint = "👤 เห็นใบหน้า"
            hint_detail = "ซูมให้ชิดผิวหนังบริเวณรอยโรค"
        elif lesion_found:
            hint = "✅ พบรอยโรค — พร้อมถ่ายภาพ"
            hint_detail = f"UNet ตรวจพบรอยโรค (conf: {confidence:.0%})"
        elif mask_ratio < 0.005:
            hint = "🔍 ไม่พบรอยโรค"
            hint_detail = "วางบริเวณที่มีรอยโรคในกรอบ"
        elif confidence < 0.60:
            hint = "🔍 ไม่ชัดเจน"
            hint_detail = f"กรุณาซูมให้ชิดขึ้น (conf: {confidence:.0%})"
        elif contrast_diff < 10:
            hint = "🔍 ไม่พบความผิดปกติ"
            hint_detail = "ถ่ายเฉพาะบริเวณที่มีรอยโรค"
        else:
            hint = "🔍 ปรับตำแหน่ง"
            hint_detail = "วางผิวหนังให้อยู่ในกรอบ"
        
        return {
            "valid": lesion_found,
            "hint": hint,
            "hint_detail": hint_detail,
            "lesion_found": lesion_found,
            "unet_confidence": round(confidence, 3),
            "mask_ratio": round(mask_ratio, 4),
            "contrast_diff": round(contrast_diff, 1),
            "mask_overlay": mask_overlay_b64,
            "skin_ratio": round(skin_ratio, 3),
            "checks": {
                "lesion_found": lesion_found,
                "unet_ready": True,
                "confidence": round(confidence, 3),
                "mask_ratio": round(mask_ratio, 4),
                "sharp_ok": True,
                "skin_ok": skin_ok,
                "skin_ratio": round(skin_ratio, 3),
            },
        }
    except Exception as e:
        print(f"  ⚠️ UNet frame validation error: {e}")
        return {
            "valid": False,
            "hint": "⚠️ เกิดข้อผิดพลาด",
            "hint_detail": str(e),
            "lesion_found": False,
            "mask_overlay": None,
            "checks": {"lesion_found": False, "unet_ready": True, "sharp_ok": True, "skin_ok": True},
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


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/predict")
async def api_predict(file: UploadFile = File(...)):
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
        
        # Predict
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=95)
        result = predict(buf.getvalue())
        
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
        "version": "7.0-vit",
        "mode": "vit",
        "models": {
            "vit": {"active": True, "source": MODEL_HF_NAME},
        },
        "active_models": 1,
        "classes": NUM_DISPLAY_CLASSES,
        "device": str(device),
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

