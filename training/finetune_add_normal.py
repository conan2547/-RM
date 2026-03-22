"""
╔═══════════════════════════════════════════════════════════════╗
║  Fine-tune Skin Cancer Model + "Normal Skin" Class           ║
║  Base: Anwarkh1/Skin_Cancer-Image_Classification (ViT)       ║
║  7 classes → 8 classes (+Normal Skin)                        ║
║  KAGGLE NOTEBOOK                                             ║
╚═══════════════════════════════════════════════════════════════╝

วิธีใช้:
  1. kaggle.com → Code → New Notebook
  2. + Add Data → search "skin cancer mnist ham10000"
     ⭐ เพิ่ม dataset: "kmader/skin-cancer-mnist-ham10000"
  3. GPU: P100 หรือ T4 x2
  4. Internet: ON (ต้องโหลดโมเดลจาก HuggingFace)
  5. วาง code ทั้งหมด → Run All
  6. ~30-60 นาที → ได้โมเดลใหม่ 8 classes!
"""

import os, sys, subprocess, random, gc, glob, time
import warnings; warnings.filterwarnings("ignore")

# ════════════════════════════════════════════════════════════════
# PHASE 0: SETUP
# ════════════════════════════════════════════════════════════════
print("="*65)
print("  PHASE 0 — SETUP")
print("="*65)

subprocess.check_call([sys.executable, "-m", "pip", "install", 
    "transformers", "datasets", "accelerate", "-q"],
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print("✅ Dependencies installed")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from transformers import AutoModelForImageClassification, AutoImageProcessor
import torchvision.transforms as T

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device: {DEVICE}")

INPUT_DIR = "/kaggle/input"
WORK_DIR = "/kaggle/working"
SAVE_DIR = f"{WORK_DIR}/models"
os.makedirs(SAVE_DIR, exist_ok=True)

# ════════════════════════════════════════════════════════════════
# PHASE 1: LOAD PRE-TRAINED MODEL
# ════════════════════════════════════════════════════════════════
print("\n"+"="*65)
print("  PHASE 1 — LOAD PRE-TRAINED MODEL")
print("="*65)

REPO_NAME = "Anwarkh1/Skin_Cancer-Image_Classification"

print(f"  📥 Loading: {REPO_NAME}")
processor = AutoImageProcessor.from_pretrained(REPO_NAME)
base_model = AutoModelForImageClassification.from_pretrained(REPO_NAME)

# Original 7 classes
ORIG_CLASSES = list(base_model.config.id2label.values())
print(f"  📋 Original classes ({len(ORIG_CLASSES)}): {ORIG_CLASSES}")

# New 8 classes (add Normal Skin)
NEW_CLASSES = ORIG_CLASSES + ["normal_skin"]
NUM_CLASSES = len(NEW_CLASSES)
print(f"  📋 New classes ({NUM_CLASSES}): {NEW_CLASSES}")

# Modify the classifier head: 7 → 8
old_classifier = base_model.classifier
in_features = old_classifier.in_features
print(f"  🔧 Replacing classifier: {in_features} → {ORIG_CLASSES.__len__()} to {in_features} → {NUM_CLASSES}")

new_classifier = nn.Linear(in_features, NUM_CLASSES)

# Copy old weights for the first 7 classes
with torch.no_grad():
    new_classifier.weight[:len(ORIG_CLASSES)] = old_classifier.weight
    new_classifier.bias[:len(ORIG_CLASSES)] = old_classifier.bias
    # Initialize new class (normal_skin) with small random weights
    nn.init.xavier_uniform_(new_classifier.weight[len(ORIG_CLASSES):].unsqueeze(0))
    new_classifier.bias[len(ORIG_CLASSES):] = 0.0

base_model.classifier = new_classifier
base_model.config.num_labels = NUM_CLASSES
base_model.config.id2label = {i: c for i, c in enumerate(NEW_CLASSES)}
base_model.config.label2id = {c: i for i, c in enumerate(NEW_CLASSES)}

base_model = base_model.to(DEVICE)
print("  ✅ Model ready with 8 classes!")


# ════════════════════════════════════════════════════════════════
# PHASE 2: PREPARE DATA
# ════════════════════════════════════════════════════════════════
print("\n"+"="*65)
print("  PHASE 2 — PREPARE DATA")
print("="*65)

# Find HAM10000 dataset
ham_dir = None
metadata_file = None

# Search for metadata CSV
for root, dirs, files in os.walk(INPUT_DIR):
    for f in files:
        if "HAM10000_metadata" in f and f.endswith(".csv"):
            metadata_file = os.path.join(root, f)
            break
    if metadata_file:
        break

if metadata_file is None:
    # Try common Kaggle paths
    candidates = [
        f"{INPUT_DIR}/skin-cancer-mnist-ham10000/HAM10000_metadata.csv",
        f"{INPUT_DIR}/skin-cancer-mnist-ham10000/HAM10000_metadata",
    ]
    for c in candidates:
        if os.path.exists(c):
            metadata_file = c
            break

if metadata_file:
    print(f"  📄 Metadata: {metadata_file}")
    df = pd.read_csv(metadata_file)
    print(f"  📊 Total images: {len(df)}")
    print(f"  📊 Classes: {df['dx'].value_counts().to_dict()}")
else:
    print("  ⚠️ HAM10000 metadata not found, searching for images...")

# Find all image files
all_images = []
for ext in ["*.jpg", "*.jpeg", "*.png"]:
    all_images.extend(glob.glob(f"{INPUT_DIR}/**/{ext}", recursive=True))
print(f"  🖼️ Total images found: {len(all_images)}")

# Build image lookup
img_lookup = {}
for p in all_images:
    basename = os.path.splitext(os.path.basename(p))[0]
    img_lookup[basename] = p

# HAM10000 class mapping to our indices
HAM_MAP = {
    "bkl": 0,  # benign_keratosis-like_lesions
    "bcc": 1,  # basal_cell_carcinoma
    "akiec": 2,  # actinic_keratoses
    "vasc": 3,  # vascular_lesions
    "nv": 4,   # melanocytic_Nevi
    "mel": 5,  # melanoma
    "df": 6,   # dermatofibroma
}

# Prepare labeled data
labeled_data = []  # (img_path, label_idx)

if metadata_file and 'dx' in df.columns:
    img_col = 'image_id' if 'image_id' in df.columns else 'image'
    for _, row in df.iterrows():
        dx = row['dx'].strip().lower()
        if dx in HAM_MAP:
            img_id = str(row[img_col]).strip()
            img_path = img_lookup.get(img_id)
            if img_path:
                labeled_data.append((img_path, HAM_MAP[dx]))
    print(f"  ✅ Labeled from metadata: {len(labeled_data)}")
else:
    # Fallback: try folder-based
    for p in all_images:
        path_lower = p.lower()
        for code, idx in HAM_MAP.items():
            if f"/{code}/" in path_lower or f"_{code}_" in path_lower:
                labeled_data.append((p, idx))
                break
    print(f"  ✅ Labeled from folders: {len(labeled_data)}")

# ── Create "Normal Skin" samples ──
# Strategy: Use patches from images with low lesion content
# OR use augmented skin-colored synthetic patches
print("\n  🧪 Creating 'Normal Skin' samples...")

NORMAL_LABEL = 7  # Index for normal_skin
normal_data = []

# Method 1: Extract skin patches from existing images
# Take corner/edge regions of dermoscopic images (usually normal skin)
random.shuffle(labeled_data)
for img_path, _ in labeled_data[:2000]:
    try:
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        # Crop corners (these are usually normal skin in dermoscopic images)
        crops = [
            img.crop((0, 0, w//3, h//3)),           # Top-left
            img.crop((2*w//3, 0, w, h//3)),         # Top-right
            img.crop((0, 2*h//3, w//3, h)),         # Bottom-left
            img.crop((2*w//3, 2*h//3, w, h)),       # Bottom-right
        ]
        # Pick one random crop
        crop = random.choice(crops)
        # Save as temp file
        tmp_path = f"{WORK_DIR}/normal_{len(normal_data):05d}.jpg"
        crop.save(tmp_path, quality=90)
        normal_data.append((tmp_path, NORMAL_LABEL))
        
        if len(normal_data) >= 1500:
            break
    except Exception:
        continue

# Method 2: Generate solid skin-colored patches
skin_colors = [
    (255, 224, 196), (245, 210, 178), (235, 195, 160),
    (225, 180, 145), (210, 165, 130), (195, 150, 115),
    (180, 135, 100), (165, 120, 85), (150, 105, 70),
    (135, 90, 60), (120, 80, 55), (100, 65, 45),
]

for i in range(500):
    base_color = random.choice(skin_colors)
    # Create with slight variation
    arr = np.zeros((224, 224, 3), dtype=np.uint8)
    for y in range(224):
        for x in range(224):
            noise = np.random.randint(-15, 15, 3)
            arr[y, x] = np.clip(np.array(base_color) + noise, 0, 255)
    
    tmp_path = f"{WORK_DIR}/skin_synth_{i:05d}.jpg"
    Image.fromarray(arr).save(tmp_path, quality=90)
    normal_data.append((tmp_path, NORMAL_LABEL))

print(f"  ✅ Normal skin samples: {len(normal_data)}")

# Combine all data
all_data = labeled_data + normal_data
random.shuffle(all_data)

print(f"\n  📊 Final dataset: {len(all_data)} images")
from collections import Counter
label_counts = Counter(label for _, label in all_data)
for idx in range(NUM_CLASSES):
    name = NEW_CLASSES[idx]
    count = label_counts.get(idx, 0)
    print(f"    [{idx}] {name:35s}: {count}")


# ════════════════════════════════════════════════════════════════
# PHASE 3: DATASET & DATALOADER
# ════════════════════════════════════════════════════════════════
print("\n"+"="*65)
print("  PHASE 3 — PREPARE DATALOADER")
print("="*65)

class SkinDataset(Dataset):
    def __init__(self, data, processor, augment=False):
        self.data = data
        self.processor = processor
        self.augment = augment
        self.aug_transform = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(30),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            T.RandomResizedCrop(224, scale=(0.8, 1.0)),
        ])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            # Return a black image on error
            img = Image.new("RGB", (224, 224), (0, 0, 0))
        
        if self.augment:
            img = self.aug_transform(img)
        
        inputs = self.processor(img, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)
        return pixel_values, label

# Split
train_data, val_data = train_test_split(all_data, test_size=0.15, 
                                         stratify=[d[1] for d in all_data],
                                         random_state=42)

train_ds = SkinDataset(train_data, processor, augment=True)
val_ds = SkinDataset(val_data, processor, augment=False)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2, drop_last=True)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2)

print(f"  Train: {len(train_data)}, Val: {len(val_data)}")


# ════════════════════════════════════════════════════════════════
# PHASE 4: FINE-TUNE
# ════════════════════════════════════════════════════════════════
print("\n"+"="*65)
print("  PHASE 4 — FINE-TUNE")
print("="*65)

# Freeze backbone, train only last layers + classifier
for name, param in base_model.named_parameters():
    if "classifier" in name or "layernorm" in name or "encoder.layer.11" in name or "encoder.layer.10" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

trainable = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
total = sum(p.numel() for p in base_model.parameters())
print(f"  📊 Trainable: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)")

# Class weights (handle imbalance)
class_counts = torch.tensor([label_counts.get(i, 1) for i in range(NUM_CLASSES)], dtype=torch.float32)
class_weights = (1.0 / class_counts)
class_weights = class_weights / class_weights.sum() * NUM_CLASSES
class_weights = class_weights.to(DEVICE)
print(f"  ⚖️ Class weights: {[f'{w:.2f}' for w in class_weights.tolist()]}")

criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, base_model.parameters()),
    lr=2e-5, weight_decay=1e-4
)

EPOCHS = 10
best_acc = 0.0
best_auc = 0.0

print(f"\n  🧠 Fine-tuning for {EPOCHS} epochs...")

for epoch in range(1, EPOCHS + 1):
    # Train
    base_model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f"  Epoch {epoch}/{EPOCHS}", leave=True)
    for pixel_values, labels in pbar:
        pixel_values = pixel_values.to(DEVICE)
        labels = torch.tensor(labels).to(DEVICE)
        
        outputs = base_model(pixel_values=pixel_values)
        loss = criterion(outputs.logits, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        preds = outputs.logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct/total:.3f}")
    
    train_acc = correct / total
    avg_loss = total_loss / len(train_loader)
    
    # Validate
    base_model.eval()
    val_preds, val_probs, val_true = [], [], []
    
    with torch.no_grad():
        for pixel_values, labels in val_loader:
            pixel_values = pixel_values.to(DEVICE)
            outputs = base_model(pixel_values=pixel_values)
            probs = torch.softmax(outputs.logits, dim=1)
            val_preds.extend(outputs.logits.argmax(1).cpu().tolist())
            val_probs.extend(probs.cpu().numpy())
            val_true.extend(labels if isinstance(labels, list) else labels.tolist())
    
    val_acc = sum(p == t for p, t in zip(val_preds, val_true)) / len(val_true)
    
    try:
        val_auc = roc_auc_score(val_true, np.array(val_probs), multi_class="ovr", average="macro")
    except:
        val_auc = 0.0
    
    print(f"  Epoch {epoch:2d} | Loss: {avg_loss:.4f} | Train Acc: {train_acc:.3f} | Val Acc: {val_acc:.3f} | AUC: {val_auc:.4f}", end="")
    
    if val_acc > best_acc:
        best_acc = val_acc
        best_auc = val_auc
        # Save model
        base_model.save_pretrained(f"{SAVE_DIR}/skin_cancer_8class")
        processor.save_pretrained(f"{SAVE_DIR}/skin_cancer_8class")
        print(f" ✓ BEST")
    else:
        print()

print(f"\n  🏆 Best Val Acc: {best_acc:.3f}, AUC: {best_auc:.4f}")


# ════════════════════════════════════════════════════════════════
# PHASE 5: EVALUATION
# ════════════════════════════════════════════════════════════════
print("\n"+"="*65)
print("  PHASE 5 — EVALUATION")
print("="*65)

# Load best model
best_model = AutoModelForImageClassification.from_pretrained(f"{SAVE_DIR}/skin_cancer_8class")
best_model = best_model.to(DEVICE).eval()

val_preds, val_probs, val_true = [], [], []
with torch.no_grad():
    for pixel_values, labels in val_loader:
        pixel_values = pixel_values.to(DEVICE)
        outputs = best_model(pixel_values=pixel_values)
        probs = torch.softmax(outputs.logits, dim=1)
        val_preds.extend(outputs.logits.argmax(1).cpu().tolist())
        val_probs.extend(probs.cpu().numpy())
        val_true.extend(labels if isinstance(labels, list) else labels.tolist())

print("\n  📊 Classification Report:")
print(classification_report(val_true, val_preds, target_names=NEW_CLASSES, digits=3))


# ════════════════════════════════════════════════════════════════
# PHASE 6: PACKAGE FOR DOWNLOAD
# ════════════════════════════════════════════════════════════════
print("\n"+"="*65)
print("  PHASE 6 — PACKAGE")
print("="*65)

# Create zip for easy download
import shutil
shutil.make_archive(f"{WORK_DIR}/skin_cancer_8class", 'zip', f"{SAVE_DIR}/skin_cancer_8class")

print(f"\n  📁 Model saved to: {SAVE_DIR}/skin_cancer_8class/")
for f in sorted(os.listdir(f"{SAVE_DIR}/skin_cancer_8class")):
    size = os.path.getsize(f"{SAVE_DIR}/skin_cancer_8class/{f}")
    s = f"{size/1024/1024:.1f} MB" if size > 1024*1024 else f"{size/1024:.1f} KB"
    print(f"    {f:30s} {s}")

print(f"""
  📦 ZIP: {WORK_DIR}/skin_cancer_8class.zip

  🎉 DONE! 
  
  📊 Results:
  ├── Best Accuracy: {best_acc:.3f}
  ├── Best AUC:      {best_auc:.4f}
  └── Classes:       {NUM_CLASSES} ({', '.join(NEW_CLASSES)})

  📥 วิธี Download:
  1. ไปที่ Output tab ด้านขวา
  2. เลือก skin_cancer_8class.zip
  3. กด Download

  🚀 วิธีใช้กับ web app:
  1. แตก zip → ใส่ folder "โมเดล/skin_cancer_8class/"
  2. แก้ app.py: REPO_NAME = "โมเดล/skin_cancer_8class"
     (เปลี่ยนจาก HuggingFace URL เป็น local path)
""")
print("═"*65)
