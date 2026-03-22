"""
╔═══════════════════════════════════════════════════════════════╗
║  Fine-tune: เพิ่ม "ผิวปกติ" (Normal Skin) เข้าโมเดล         ║
║  Base: Skin_Cancer-Image_Classification (ViT, 7 classes)     ║
║  Result: 8 classes (7 โรค + ผิวปกติ)                         ║
║  รันบน Mac (MPS GPU) ได้เลย!                                 ║
╚═══════════════════════════════════════════════════════════════╝
"""
import os, sys, subprocess, random, time, shutil
import warnings; warnings.filterwarnings("ignore")

# ════════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════════
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
MODEL_DIR = os.path.join(PROJECT_DIR, "Skin_Cancer-Image_Classification")
NORMAL_DIR = os.path.join(PROJECT_DIR, "normal_skin_data")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "model_8class")

BATCH_SIZE = 8   # เล็กหน่อยสำหรับ Mac
EPOCHS = 10
LR = 2e-5

print("="*60)
print("  Fine-tune Skin Cancer Model + Normal Skin")
print("="*60)
print(f"  Model: {MODEL_DIR}")
print(f"  Output: {OUTPUT_DIR}")

# ════════════════════════════════════════════════════════════════
# PHASE 1: INSTALL & IMPORT
# ════════════════════════════════════════════════════════════════
print("\n📦 Installing dependencies...")
subprocess.check_call([sys.executable, "-m", "pip", "install", 
    "transformers", "datasets", "accelerate", "Pillow", 
    "scikit-learn", "tqdm", "-q"],
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from transformers import AutoModelForImageClassification, AutoImageProcessor
import torchvision.transforms as T

# Device
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("  ✅ Using Apple MPS (GPU)")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("  ✅ Using CUDA GPU")
else:
    DEVICE = torch.device("cpu")
    print("  ⚠️ Using CPU (slow)")

# ════════════════════════════════════════════════════════════════
# PHASE 2: DOWNLOAD NORMAL SKIN IMAGES
# ════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  PHASE 2 — DOWNLOAD NORMAL SKIN DATA")
print("="*60)

os.makedirs(NORMAL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Download from HuggingFace datasets
print("  📥 Downloading skin dataset from HuggingFace...")
try:
    from datasets import load_dataset
    
    # Load marmal88/skin_cancer dataset (same as training data)
    ds = load_dataset("marmal88/skin_cancer", split="train", trust_remote_code=True)
    print(f"  ✅ Dataset loaded: {len(ds)} images")
    
    # Save disease images to temp folders by label
    disease_dirs = {}
    label_names = ds.features["dx"].names if hasattr(ds.features["dx"], "names") else None
    
    DISEASE_DIR = os.path.join(PROJECT_DIR, "disease_data_tmp")
    os.makedirs(DISEASE_DIR, exist_ok=True)
    
    print("  📁 Organizing disease images...")
    disease_counts = {}
    
    for i, sample in enumerate(tqdm(ds, desc="  Saving images")):
        img = sample["image"]
        label = sample["dx"]
        
        # Get label name
        if label_names:
            label_name = label_names[label] if isinstance(label, int) else str(label)
        else:
            label_name = str(label)
        
        label_dir = os.path.join(DISEASE_DIR, label_name)
        os.makedirs(label_dir, exist_ok=True)
        
        img_path = os.path.join(label_dir, f"{i:05d}.jpg")
        if isinstance(img, Image.Image):
            img.save(img_path, quality=90)
        
        disease_counts[label_name] = disease_counts.get(label_name, 0) + 1
    
    print(f"\n  📊 Disease image counts:")
    for name, count in sorted(disease_counts.items()):
        print(f"    {name:35s}: {count}")
    
    HAS_DISEASE_DATA = True

except Exception as e:
    print(f"  ⚠️ Could not load HuggingFace dataset: {e}")
    print("  📌 Will use only normal skin images + existing model")
    HAS_DISEASE_DATA = False
    DISEASE_DIR = None

# ── Generate Normal Skin Images ──
print("\n  🧪 Creating normal skin images...")

normal_count = 0

# Method 1: Crop edges from disease images (edges = normal skin usually)
if HAS_DISEASE_DATA and DISEASE_DIR:
    all_disease_imgs = []
    for root, dirs, files in os.walk(DISEASE_DIR):
        for f in files:
            if f.endswith(".jpg"):
                all_disease_imgs.append(os.path.join(root, f))
    
    random.shuffle(all_disease_imgs)
    
    for img_path in all_disease_imgs[:1500]:
        try:
            img = Image.open(img_path).convert("RGB")
            w, h = img.size
            
            # Crop corners (normal skin area in dermoscopic images)
            crop_size_w = w // 4
            crop_size_h = h // 4
            
            corners = [
                (0, 0, crop_size_w, crop_size_h),
                (w - crop_size_w, 0, w, crop_size_h),
                (0, h - crop_size_h, crop_size_w, h),
                (w - crop_size_w, h - crop_size_h, w, h),
            ]
            
            crop = img.crop(random.choice(corners))
            crop = crop.resize((224, 224), Image.LANCZOS)
            
            save_path = os.path.join(NORMAL_DIR, f"crop_{normal_count:05d}.jpg")
            crop.save(save_path, quality=90)
            normal_count += 1
            
        except Exception:
            continue

# Method 2: Synthetic skin patches
print("  🎨 Generating synthetic skin patches...")
skin_tones = [
    # Light skin
    (255, 224, 196), (250, 218, 190), (245, 210, 178),
    (240, 205, 170), (235, 198, 162), (230, 190, 155),
    # Medium skin
    (225, 180, 145), (215, 170, 135), (205, 160, 125),
    (195, 150, 115), (185, 140, 105), (175, 130, 95),
    # Dark skin
    (165, 120, 85), (150, 105, 70), (135, 90, 60),
    (120, 80, 55), (100, 65, 45), (85, 55, 35),
]

for i in range(500):
    base_color = np.array(random.choice(skin_tones), dtype=np.float32)
    
    # Create natural skin texture
    arr = np.zeros((224, 224, 3), dtype=np.float32)
    
    # Add gradient (natural skin has subtle gradients)
    for y in range(224):
        for x in range(224):
            # Base color with gradient
            gradient = (y / 224 - 0.5) * 10 + (x / 224 - 0.5) * 5
            noise = np.random.normal(0, 8, 3)
            pixel = base_color + gradient + noise
            arr[y, x] = np.clip(pixel, 0, 255)
    
    arr = arr.astype(np.uint8)
    
    # Add subtle texture (pores, fine lines)
    for _ in range(random.randint(50, 200)):
        cx, cy = random.randint(0, 223), random.randint(0, 223)
        r = random.randint(1, 3)
        darkness = random.randint(-20, -5)
        y_min = max(0, cy - r)
        y_max = min(224, cy + r)
        x_min = max(0, cx - r)
        x_max = min(224, cx + r)
        arr[y_min:y_max, x_min:x_max] = np.clip(
            arr[y_min:y_max, x_min:x_max].astype(np.int16) + darkness, 0, 255
        ).astype(np.uint8)
    
    save_path = os.path.join(NORMAL_DIR, f"synth_{normal_count:05d}.jpg")
    Image.fromarray(arr).save(save_path, quality=90)
    normal_count += 1

print(f"  ✅ Normal skin images: {normal_count}")


# ════════════════════════════════════════════════════════════════
# PHASE 3: PREPARE DATASET
# ════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  PHASE 3 — PREPARE DATASET")
print("="*60)

# Load processor
processor = AutoImageProcessor.from_pretrained(MODEL_DIR)

# Class mapping
ORIG_CLASSES = [
    "benign_keratosis-like_lesions", "basal_cell_carcinoma",
    "actinic_keratoses", "vascular_lesions", "melanocytic_Nevi",
    "melanoma", "dermatofibroma"
]
NEW_CLASSES = ORIG_CLASSES + ["normal_skin"]
NUM_CLASSES = len(NEW_CLASSES)

# Collect all data: (image_path, label_idx)
all_data = []

# Disease images
if HAS_DISEASE_DATA and DISEASE_DIR:
    # Map disease folder names to our class indices
    folder_to_idx = {}
    for folder_name in os.listdir(DISEASE_DIR):
        folder_lower = folder_name.lower()
        for idx, cls in enumerate(ORIG_CLASSES):
            if cls.lower() == folder_lower or folder_lower in cls.lower() or cls.lower() in folder_lower:
                folder_to_idx[folder_name] = idx
                break
        # HAM10000 short codes
        if folder_name not in folder_to_idx:
            ham_map = {"bkl": 0, "bcc": 1, "akiec": 2, "vasc": 3, "nv": 4, "mel": 5, "df": 6}
            if folder_lower in ham_map:
                folder_to_idx[folder_name] = ham_map[folder_lower]
    
    print(f"  📋 Folder mapping: {folder_to_idx}")
    
    for folder_name, idx in folder_to_idx.items():
        folder_path = os.path.join(DISEASE_DIR, folder_name)
        for f in os.listdir(folder_path):
            if f.endswith((".jpg", ".png")):
                all_data.append((os.path.join(folder_path, f), idx))

# Normal skin images (class 7)
for f in os.listdir(NORMAL_DIR):
    if f.endswith((".jpg", ".png")):
        all_data.append((os.path.join(NORMAL_DIR, f), 7))

random.shuffle(all_data)

# Count per class
from collections import Counter
label_counts = Counter(label for _, label in all_data)
print(f"\n  📊 Dataset: {len(all_data)} images")
for idx in range(NUM_CLASSES):
    name = NEW_CLASSES[idx]
    count = label_counts.get(idx, 0)
    marker = "🆕" if idx == 7 else "  "
    print(f"  {marker} [{idx}] {name:35s}: {count}")

# Balance: undersample majority classes to max 2000
MAX_PER_CLASS = 2000
balanced_data = []
for idx in range(NUM_CLASSES):
    class_items = [(p, l) for p, l in all_data if l == idx]
    if len(class_items) > MAX_PER_CLASS:
        class_items = random.sample(class_items, MAX_PER_CLASS)
    balanced_data.extend(class_items)

random.shuffle(balanced_data)
print(f"\n  📊 Balanced dataset: {len(balanced_data)} images")

# Split
train_data, val_data = train_test_split(
    balanced_data, test_size=0.15,
    stratify=[d[1] for d in balanced_data],
    random_state=42
)
print(f"  Train: {len(train_data)}, Val: {len(val_data)}")


# ── Dataset Class ──
class SkinDataset(Dataset):
    def __init__(self, data, processor, augment=False):
        self.data = data
        self.processor = processor
        self.augment = augment
        self.aug_transform = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(20),
            T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
            T.RandomResizedCrop(224, scale=(0.85, 1.0)),
        ])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        try:
            img = Image.open(img_path).convert("RGB")
        except:
            img = Image.new("RGB", (224, 224))
        
        if self.augment:
            img = self.aug_transform(img)
        
        inputs = self.processor(img, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)
        return pixel_values, label

train_ds = SkinDataset(train_data, processor, augment=True)
val_ds = SkinDataset(val_data, processor, augment=False)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


# ════════════════════════════════════════════════════════════════
# PHASE 4: MODIFY MODEL (7 → 8 classes)
# ════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  PHASE 4 — MODIFY MODEL")
print("="*60)

model = AutoModelForImageClassification.from_pretrained(MODEL_DIR)

# Replace classifier: 7 → 8
old_classifier = model.classifier
in_features = old_classifier.in_features

new_classifier = nn.Linear(in_features, NUM_CLASSES)
with torch.no_grad():
    new_classifier.weight[:7] = old_classifier.weight
    new_classifier.bias[:7] = old_classifier.bias
    nn.init.xavier_uniform_(new_classifier.weight[7:].unsqueeze(0))
    new_classifier.bias[7:] = 0.0

model.classifier = new_classifier
model.config.num_labels = NUM_CLASSES
model.config.id2label = {i: c for i, c in enumerate(NEW_CLASSES)}
model.config.label2id = {c: i for i, c in enumerate(NEW_CLASSES)}

model = model.to(DEVICE)
print(f"  ✅ Classifier: {in_features} → {NUM_CLASSES} classes")

# Freeze backbone, train last layers + classifier
for name, param in model.named_parameters():
    if any(key in name for key in ["classifier", "layernorm", "encoder.layer.11", "encoder.layer.10"]):
        param.requires_grad = True
    else:
        param.requires_grad = False

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"  📊 Trainable: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)")


# ════════════════════════════════════════════════════════════════
# PHASE 5: TRAIN
# ════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  PHASE 5 — TRAINING")
print("="*60)

class_weights = torch.tensor(
    [1.0 / max(label_counts.get(i, 1), 1) for i in range(NUM_CLASSES)],
    dtype=torch.float32
)
class_weights = class_weights / class_weights.sum() * NUM_CLASSES
class_weights = class_weights.to(DEVICE)

criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR, weight_decay=1e-4
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

best_acc = 0.0

for epoch in range(1, EPOCHS + 1):
    # Train
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f"  Epoch {epoch}/{EPOCHS}")
    for pixel_values, labels in pbar:
        pixel_values = pixel_values.to(DEVICE)
        labels = torch.tensor(labels, dtype=torch.long).to(DEVICE)
        
        outputs = model(pixel_values=pixel_values)
        loss = criterion(outputs.logits, labels)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        preds = outputs.logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        pbar.set_postfix(loss=f"{loss.item():.3f}", acc=f"{correct/total:.3f}")
    
    scheduler.step()
    train_acc = correct / total
    
    # Validate
    model.eval()
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for pixel_values, labels in val_loader:
            pixel_values = pixel_values.to(DEVICE)
            labels = torch.tensor(labels, dtype=torch.long).to(DEVICE)
            
            outputs = model(pixel_values=pixel_values)
            preds = outputs.logits.argmax(dim=1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)
    
    val_acc = val_correct / val_total
    avg_loss = total_loss / len(train_loader)
    
    saved = ""
    if val_acc > best_acc:
        best_acc = val_acc
        model.save_pretrained(OUTPUT_DIR)
        processor.save_pretrained(OUTPUT_DIR)
        saved = " ✓ SAVED"
    
    print(f"  Epoch {epoch:2d} | Loss: {avg_loss:.4f} | Train: {train_acc:.3f} | Val: {val_acc:.3f}{saved}")

print(f"\n  🏆 Best Val Accuracy: {best_acc:.3f}")


# ════════════════════════════════════════════════════════════════
# PHASE 6: DONE
# ════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  PHASE 6 — COMPLETE!")
print("="*60)

print(f"""
  ✅ โมเดลใหม่ 8 classes บันทึกแล้วที่:
  📁 {OUTPUT_DIR}/

  📊 Classes:
""")
for i, cls in enumerate(NEW_CLASSES):
    marker = "🆕" if i == 7 else "  "
    print(f"    {marker} [{i}] {cls}")

print(f"""
  🏆 Best Accuracy: {best_acc:.3f}

  🚀 เพื่อใช้กับ web app ให้แก้ app.py:
     เปลี่ยน REPO_NAME เป็น:
     REPO_NAME = "{OUTPUT_DIR}"

  🧹 หลังเทรนเสร็จสามารถลบ folder temp ได้:
     rm -rf "{NORMAL_DIR}"
     rm -rf "{DISEASE_DIR if DISEASE_DIR else ''}"
""")

# Cleanup temp files
print("  🧹 Cleaning up temp data...")
if os.path.exists(NORMAL_DIR):
    shutil.rmtree(NORMAL_DIR, ignore_errors=True)
if DISEASE_DIR and os.path.exists(DISEASE_DIR):
    shutil.rmtree(DISEASE_DIR, ignore_errors=True)
    
print("  ✅ Done!")
print("="*60)
