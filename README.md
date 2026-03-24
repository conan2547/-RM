---
title: Skin Cancer Detection
emoji: 🔬
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# 🔬 AI for Skin Cancer Detection

ระบบตรวจสอบโรคผิวหนังด้วย AI โดยใช้ Vision Transformer (ViT) สำหรับจำแนกรอยโรคผิวหนัง 6 ประเภท

## 📁 Project Structure

```
โปรเจค RM/
├── models/                  # 🧠 AI Model Weights
│   ├── vit_8class/          # ViT classification model (HuggingFace)
│   ├── dinov2/              # DinoV2 feature extraction model
│   └── lora/                # LoRA fine-tuned weights
│
├── training/                # 🔬 Training Scripts
│   ├── finetune_add_normal.py       # Fine-tune with normal skin class
│   ├── train_normal_only.py         # Train normal skin classifier
│   ├── train_normal_skin.py         # Normal skin training pipeline
│   ├── train_dinov2_kaggle.py       # DinoV2 training on Kaggle
│   └── train_dinov2_kaggle.ipynb    # Notebook version
│
├── data/                    # 📊 Training Data
│   └── normal_skin/         # Normal skin dataset (crop + synthetic)
│
├── web/                     # 🌐 Web Application (FastAPI)
│   ├── app.py               # Main backend server
│   ├── requirements.txt     # Python dependencies
│   ├── static/images/       # Static assets (logo, disease images)
│   └── templates/           # HTML templates
│       └── index.html       # Main web interface
│
├── .gitignore               # Git ignore rules  
└── README.md                # This file
```

## 🎯 Supported Skin Conditions

| # | Condition (EN) | ชื่อภาษาไทย | Risk Level |
|---|---------------|-------------|------------|
| 1 | Melanoma | เมลาโนมา | 🔴 Critical |
| 2 | Basal Cell Carcinoma (BCC) | มะเร็งเบเซลเซลล์ | 🟠 High |
| 3 | Squamous Cell Carcinoma (SCC) | มะเร็งสความัสเซลล์ | 🟠 High |
| 4 | Melanocytic Nevi | ไฝ / ขี้แมลงวัน | 🟢 Low |
| 5 | Seborrheic Keratosis | กระเนื้อ | 🟢 Low |
| 6 | Normal Skin | ผิวหนังปกติ | ✅ Safe |

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- pip

### Installation

```bash
# 1. Clone the project
cd web/

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the server
uvicorn app:app --host 0.0.0.0 --port 8000
```

### Access
Open browser at: **http://localhost:8000**

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Backend | FastAPI (Python) |
| AI Model | Vision Transformer (ViT) via HuggingFace |
| Segmentation | UNet |
| Frontend | HTML + Tailwind CSS + Vanilla JS |
| Model Hub | HuggingFace (`bnmbanhmi/seekwell_skincancer_v2`) |

## 📝 Development Notes

- โมเดลจะดาวน์โหลดอัตโนมัติจาก HuggingFace Hub เมื่อ run ครั้งแรก
- ไฟล์โมเดล `.safetensors` และ `.pt` ถูก ignore จาก git (ขนาดใหญ่)
- Training data อยู่ใน `data/` ซึ่งถูก ignore จาก git เช่นกัน

## 👥 Contributors

พัฒนาโดยทีม RM
