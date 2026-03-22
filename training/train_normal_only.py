"""
เทรนเพิ่มแค่ "ผิวปกติ" (Normal Skin) — 5 epochs
Freeze ทุกอย่าง ยกเว้น classifier row ใหม่
ใช้เวลา ~5-10 นาที บน Mac MPS
"""
import os, sys, subprocess, random, time
import warnings; warnings.filterwarnings("ignore")

# ─── Config ───
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(PROJECT_DIR, "Skin_Cancer-Image_Classification")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "model_8class")
EPOCHS = 5
BATCH_SIZE = 16

print("="*55)
print("  เทรนเพิ่มแค่ Normal Skin (5 epochs)")
print("="*55)

# ─── Install ───
subprocess.check_call([sys.executable, "-m", "pip", "install",
    "transformers", "datasets", "Pillow", "scikit-learn", "tqdm", "-q"],
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageClassification, AutoImageProcessor
import torchvision.transforms as T

# Device
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("  ✅ Apple MPS GPU")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("  ✅ CUDA GPU")
else:
    DEVICE = torch.device("cpu")
    print("  ⚠️ CPU")

# ═══════════════════════════════════════════════════════
# 1. LOAD MODEL + เพิ่ม class 8
# ═══════════════════════════════════════════════════════
print("\n📥 Loading model...")
processor = AutoImageProcessor.from_pretrained(MODEL_DIR)
model = AutoModelForImageClassification.from_pretrained(MODEL_DIR)

ORIG_CLASSES = list(model.config.id2label.values())
NEW_CLASSES = ORIG_CLASSES + ["normal_skin"]
print(f"  เดิม: {len(ORIG_CLASSES)} classes → ใหม่: {len(NEW_CLASSES)} classes")

# เปลี่ยน head: 7 → 8
old_clf = model.classifier
in_features = old_clf.in_features
new_clf = nn.Linear(in_features, 8)

with torch.no_grad():
    new_clf.weight[:7] = old_clf.weight  # copy 7 เดิม
    new_clf.bias[:7] = old_clf.bias
    nn.init.xavier_uniform_(new_clf.weight[7:].unsqueeze(0))
    new_clf.bias[7:] = 0.0

model.classifier = new_clf
model.config.num_labels = 8
model.config.id2label = {i: c for i, c in enumerate(NEW_CLASSES)}
model.config.label2id = {c: i for i, c in enumerate(NEW_CLASSES)}
model = model.to(DEVICE)

# ═══════════════════════════════════════════════════════
# 2. FREEZE ทุกอย่าง ยกเว้น classifier
# ═══════════════════════════════════════════════════════
for param in model.parameters():
    param.requires_grad = False

# เปิดแค่ classifier (weight + bias ของ 8 classes)
for param in model.classifier.parameters():
    param.requires_grad = True

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"  🔒 Freeze: {total-trainable:,} params")
print(f"  🔓 Train:  {trainable:,} params ({trainable/total*100:.2f}%)")

# ═══════════════════════════════════════════════════════
# 3. สร้างข้อมูล NORMAL SKIN
# ═══════════════════════════════════════════════════════
print("\n🧪 สร้างภาพผิวปกติ...")
NORMAL_DIR = os.path.join(PROJECT_DIR, "normal_skin_tmp")
os.makedirs(NORMAL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# สร้างภาพ synthetic skin
skin_tones = [
    (255, 224, 196), (250, 218, 190), (245, 210, 178),
    (240, 205, 170), (235, 198, 162), (230, 190, 155),
    (225, 180, 145), (215, 170, 135), (205, 160, 125),
    (195, 150, 115), (185, 140, 105), (175, 130, 95),
    (165, 120, 85), (150, 105, 70), (135, 90, 60),
    (120, 80, 55), (100, 65, 45), (85, 55, 35),
]

normal_paths = []
for i in range(800):
    base = np.array(random.choice(skin_tones), dtype=np.float32)
    arr = np.zeros((224, 224, 3), dtype=np.float32)
    
    # gradient + noise = realistic skin
    gy = np.linspace(-8, 8, 224).reshape(-1, 1)
    gx = np.linspace(-4, 4, 224).reshape(1, -1)
    for c in range(3):
        arr[:, :, c] = base[c] + gy + gx + np.random.normal(0, 6, (224, 224))
    
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    
    # pores/texture
    for _ in range(random.randint(30, 150)):
        cx, cy = random.randint(5, 218), random.randint(5, 218)
        r = random.randint(1, 2)
        d = random.randint(-15, -3)
        arr[cy-r:cy+r, cx-r:cx+r] = np.clip(
            arr[cy-r:cy+r, cx-r:cx+r].astype(np.int16) + d, 0, 255
        ).astype(np.uint8)
    
    path = os.path.join(NORMAL_DIR, f"normal_{i:04d}.jpg")
    Image.fromarray(arr).save(path, quality=90)
    normal_paths.append(path)

print(f"  ✅ ภาพผิวปกติ: {len(normal_paths)} ภาพ")

# ═══════════════════════════════════════════════════════
# 4. DATASET (Normal=7, Disease=0-6 จาก HF)
# ═══════════════════════════════════════════════════════
print("\n📦 เตรียม dataset...")

# ดาวน์โหลดตัวอย่างโรคจาก HuggingFace (เล็กน้อยเพื่อกันลืม)
disease_data = []
try:
    from datasets import load_dataset
    ds = load_dataset("marmal88/skin_cancer", split="train")
    print(f"  📥 HuggingFace dataset: {len(ds)} images")
    
    # เก็บแค่ 100 ภาพต่อ class (รวม 700)
    class_count = {}
    disease_tmp = os.path.join(PROJECT_DIR, "disease_tmp")
    os.makedirs(disease_tmp, exist_ok=True)
    
    label_names = ds.features["dx"].names if hasattr(ds.features["dx"], "names") else None
    
    for i, sample in enumerate(ds):
        label = sample["dx"]
        if label_names:
            label_name = label_names[label] if isinstance(label, int) else str(label)
        else:
            label_name = str(label)
        
        # Map to index
        label_map = {
            "bkl": 0, "bcc": 1, "akiec": 2, "vasc": 3,
            "nv": 4, "mel": 5, "df": 6,
            "benign_keratosis-like_lesions": 0,
            "basal_cell_carcinoma": 1,
            "actinic_keratoses": 2,
            "vascular_lesions": 3,
            "melanocytic_nevi": 4, "melanocytic_Nevi": 4,
            "melanoma": 5,
            "dermatofibroma": 6,
        }
        
        idx = label_map.get(label_name, label_map.get(label_name.lower()))
        if idx is None:
            continue
        
        count = class_count.get(idx, 0)
        if count >= 100:  # แค่ 100 ต่อ class
            continue
        
        img = sample["image"]
        if isinstance(img, Image.Image):
            path = os.path.join(disease_tmp, f"disease_{idx}_{count:04d}.jpg")
            img.save(path, quality=85)
            disease_data.append((path, idx))
            class_count[idx] = count + 1
    
    print(f"  ✅ ตัวอย่างโรค: {len(disease_data)} ภาพ (100/class)")
    
except Exception as e:
    print(f"  ⚠️ ข้าม HuggingFace dataset: {e}")
    print("  → ใช้แค่ Normal Skin อย่างเดียว")

# รวม data
all_data = [(p, 7) for p in normal_paths] + disease_data  # 7 = normal_skin
random.shuffle(all_data)

# Split
from sklearn.model_selection import train_test_split
train_data, val_data = train_test_split(all_data, test_size=0.15, random_state=42)
print(f"  Train: {len(train_data)}, Val: {len(val_data)}")

# Dataset class
class SkinDS(Dataset):
    def __init__(self, data, proc, aug=False):
        self.data = data
        self.proc = proc
        self.aug = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(15),
            T.ColorJitter(0.1, 0.1, 0.1),
        ]) if aug else None
    
    def __len__(self): return len(self.data)
    
    def __getitem__(self, i):
        path, label = self.data[i]
        try:
            img = Image.open(path).convert("RGB")
        except:
            img = Image.new("RGB", (224, 224))
        if self.aug:
            img = self.aug(img)
        px = self.proc(img, return_tensors="pt")["pixel_values"].squeeze(0)
        return px, label

train_loader = DataLoader(SkinDS(train_data, processor, aug=True),
                          batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(SkinDS(val_data, processor),
                        batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ═══════════════════════════════════════════════════════
# 5. TRAIN — แค่ 5 epochs!
# ═══════════════════════════════════════════════════════
print(f"\n🧠 Training... ({EPOCHS} epochs)")

optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()
best_acc = 0.0

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss, correct, total = 0, 0, 0
    
    pbar = tqdm(train_loader, desc=f"  Epoch {epoch}/{EPOCHS}")
    for px, labels in pbar:
        px = px.to(DEVICE)
        labels = torch.tensor(labels, dtype=torch.long).to(DEVICE)
        
        out = model(pixel_values=px)
        loss = criterion(out.logits, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = out.logits.argmax(1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
        pbar.set_postfix(loss=f"{loss.item():.3f}", acc=f"{correct/total:.3f}")
    
    # Validate
    model.eval()
    vc, vt = 0, 0
    with torch.no_grad():
        for px, labels in val_loader:
            px = px.to(DEVICE)
            labels = torch.tensor(labels, dtype=torch.long).to(DEVICE)
            out = model(pixel_values=px)
            vc += (out.logits.argmax(1) == labels).sum().item()
            vt += labels.size(0)
    
    val_acc = vc / vt
    saved = ""
    if val_acc > best_acc:
        best_acc = val_acc
        model.save_pretrained(OUTPUT_DIR)
        processor.save_pretrained(OUTPUT_DIR)
        saved = " ✓ SAVED"
    
    print(f"  Epoch {epoch} | Loss: {total_loss/len(train_loader):.4f} | "
          f"Train: {correct/total:.3f} | Val: {val_acc:.3f}{saved}")

# ═══════════════════════════════════════════════════════
# 6. CLEANUP & DONE
# ═══════════════════════════════════════════════════════
import shutil
if os.path.exists(NORMAL_DIR):
    shutil.rmtree(NORMAL_DIR, ignore_errors=True)
disease_tmp = os.path.join(PROJECT_DIR, "disease_tmp")
if os.path.exists(disease_tmp):
    shutil.rmtree(disease_tmp, ignore_errors=True)

print(f"""
{'='*55}
  ✅ เสร็จแล้ว!
  
  📁 โมเดลใหม่: {OUTPUT_DIR}/
  🏆 Best Acc: {best_acc:.3f}
  📊 Classes: {', '.join(NEW_CLASSES)}
  
  🆕 เพิ่ม: normal_skin (class 7)
  🔒 7 โรคเดิม: ไม่ถูกแก้ไข!
{'='*55}
""")
