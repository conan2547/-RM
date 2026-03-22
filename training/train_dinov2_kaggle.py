"""
╔══════════════════════════════════════════════════════════════════╗
║  🧬 DINOv2 Skin Cancer Classification — Kaggle Training Script ║
║  Fine-tune DINOv2 (facebook/dinov2-base) on ISIC 2019 dataset  ║
║  5 Classes: MEL, BCC, SCC, DF, NV                               ║
╚══════════════════════════════════════════════════════════════════╝

📋 วิธีใช้บน Kaggle:
1. สร้าง Notebook ใหม่บน Kaggle
2. เปิด GPU (Settings → Accelerator → GPU P100/T4 x2)
3. เพิ่ม Dataset: "nroman/2019-isic-challenge"  ← ISIC 2019 (มี SCC!)
4. Copy โค้ดนี้ลงไปทั้งหมด → Run All
5. โมเดลจะถูก save ไว้ที่ /kaggle/working/dinov2-skin-cancer-5class/
6. ดาวน์โหลด model.safetensors + config.json กลับมาใช้งาน

📊 5 คลาสที่เทรน:
  [0] melanoma                  — เมลาโนมา (มะเร็งร้ายแรงที่สุด)
  [1] basal_cell_carcinoma      — มะเร็งเซลล์ฐาน (BCC)
  [2] squamous_cell_carcinoma   — มะเร็งเซลล์สความัส (SCC) ← ตัวจริง!
  [3] dermatofibroma            — เนื้องอกไม่ร้าย
  [4] melanocytic_Nevi          — ไฝ
"""

# ═══════════════════════════════════════════════════
#  📦 CELL 1: Install Dependencies
# ═══════════════════════════════════════════════════

!pip install -q transformers datasets evaluate accelerate scikit-learn safetensors pillow

# ── เช็ค GPU Compatibility ──
# PyTorch 2.5+ ไม่รองรับ P100 (sm_60) ต้องดาวน์เกรด
import torch
if torch.cuda.is_available():
    cc = torch.cuda.get_device_capability(0)
    gpu_name = torch.cuda.get_device_name(0)
    print(f"  🖥️  GPU: {gpu_name} (compute capability: {cc[0]}.{cc[1]})")
    if cc[0] < 7:  # sm_60/sm_61 = P100, K80 etc.
        print(f"  ⚠️  GPU compute capability {cc[0]}.{cc[1]} < 7.0")
        print(f"  📋 ทางเลือก 2 ทาง:")
        print(f"     1. [แนะนำ] ไปที่ Settings → Accelerator → เปลี่ยนเป็น GPU T4 x2")
        print(f"     2. ดาวน์เกรด PyTorch (กำลังทำอัตโนมัติ...)")
        print(f"  🔄 Downgrading PyTorch for P100 compatibility...")
        !pip install -q torch==2.4.1 torchvision==0.19.1
        # Reimport after downgrade
        import importlib
        importlib.reload(torch)
        print(f"  ✅ PyTorch {torch.__version__} installed (P100 compatible)")
    else:
        print(f"  ✅ GPU supported! (compute capability >= 7.0)")

import os
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings("ignore")

print("=" * 65)
print("  🧬 DINOv2 Skin Cancer Classifier — Training Script")
print("  📋 5 Classes | ISIC 2019 Dataset (มี SCC จริง!) | Kaggle GPU")
print("=" * 65)

# Check GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(f"  🖥️  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("  ⚠️  No GPU detected! Training will be very slow.")
print()


# ═══════════════════════════════════════════════════
#  📦 CELL 2: Configuration
# ═══════════════════════════════════════════════════

class Config:
    # ── Model ──
    MODEL_NAME = "facebook/dinov2-base"          # DINOv2 Base (ViT-B/14)
    NUM_CLASSES = 5
    
    # ── Training Hyperparameters ──
    LEARNING_RATE = 2e-5
    TRAIN_BATCH_SIZE = 64
    EVAL_BATCH_SIZE = 16
    SEED = 42
    NUM_EPOCHS = 15
    WARMUP_RATIO = 0.1
    WEIGHT_DECAY = 0.01
    
    # ── Data ──
    IMAGE_SIZE = 224
    
    # ── ISIC 2019: 5 คลาสที่ต้องการ ──
    # ISIC 2019 มี 9 คลาส เราเลือกเอา 5 ที่ต้องการ
    # ตัดออก: AK (akiec), BKL (benign keratosis), VASC, UNK
    CLASS_NAMES = [
        "melanoma",                       # MEL  — เมลาโนมา
        "basal_cell_carcinoma",           # BCC  — มะเร็งเซลล์ฐาน
        "squamous_cell_carcinoma",        # SCC  — มะเร็งเซลล์สความัส (ตัวจริง!)
        "dermatofibroma",                 # DF   — เนื้องอกไม่ร้าย
        "melanocytic_Nevi",               # NV   — ไฝ
    ]
    
    # ISIC 2019 label column names → our index
    ISIC_TO_IDX = {
        "MEL":  0,   # melanoma
        "BCC":  1,   # basal_cell_carcinoma
        "SCC":  2,   # squamous_cell_carcinoma  ← SCC จริง!
        "DF":   3,   # dermatofibroma
        "NV":   4,   # melanocytic_Nevi
    }
    
    # Classes to KEEP from ISIC 2019
    KEEP_CLASSES = ["MEL", "BCC", "SCC", "DF", "NV"]
    
    # ── Paths (Kaggle) ──
    # Dataset: "nroman/2019-isic-challenge" on Kaggle
    KAGGLE_DATA = Path("/kaggle/input/2019-isic-challenge")
    OUTPUT_DIR = Path("/kaggle/working/dinov2-skin-cancer-5class")
    
    # ── Class Weights for Imbalanced Data ──
    # ISIC 2019 approx: NV≈12,875, MEL≈4,522, BCC≈3,323, SCC≈628, DF≈239
    USE_CLASS_WEIGHTS = True


config = Config()

# Set seed for reproducibility
torch.manual_seed(config.SEED)
np.random.seed(config.SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(config.SEED)

print(f"  📋 Model: {config.MODEL_NAME}")
print(f"  🎯 Classes: {config.NUM_CLASSES}")
for i, name in enumerate(config.CLASS_NAMES):
    print(f"     [{i}] {name}")
print(f"  📊 Epochs: {config.NUM_EPOCHS}")
print(f"  📊 LR: {config.LEARNING_RATE}")
print(f"  📊 Batch: {config.TRAIN_BATCH_SIZE}")
print()


# ═══════════════════════════════════════════════════
#  📦 CELL 3: Load & Prepare ISIC 2019 Dataset
# ═══════════════════════════════════════════════════

print("📂 Loading ISIC 2019 dataset...")

# ── Find the ground truth CSV ──
# ISIC 2019 format: CSV with columns [image, MEL, NV, BCC, AK, BKL, DF, VASC, SCC, UNK]
# Each row has 1 in the class column, 0 in others (one-hot encoded)
gt_path = None
possible_gt_paths = [
    config.KAGGLE_DATA / "ISIC_2019_Training_GroundTruth.csv",
    config.KAGGLE_DATA / "ISIC2019" / "ISIC_2019_Training_GroundTruth.csv",
    Path("/kaggle/input/isic-2019-skin-lesion-images-for-classification/ISIC_2019_Training_GroundTruth.csv"),
    Path("/kaggle/input/skin-lesion-images-for-classification/ISIC_2019_Training_GroundTruth.csv"),
]

for candidate in possible_gt_paths:
    if candidate.exists():
        gt_path = candidate
        break

# If not found, search recursively
if gt_path is None:
    import glob
    csvs = glob.glob("/kaggle/input/**/*GroundTruth*.csv", recursive=True)
    if not csvs:
        csvs = glob.glob("/kaggle/input/**/*.csv", recursive=True)
    print(f"  🔍 Found CSVs: {csvs[:10]}")
    for csv in csvs:
        if "groundtruth" in csv.lower() or "ground_truth" in csv.lower() or "training" in csv.lower():
            gt_path = Path(csv)
            break
    if gt_path is None and csvs:
        gt_path = Path(csvs[0])

assert gt_path is not None, """
❌ Cannot find ISIC 2019 Ground Truth CSV!
Please add the dataset on Kaggle:
  1. Go to "Add Data" → Search "2019 isic challenge" or "isic 2019"
  2. Add dataset by "nroman" or similar
  3. Re-run this notebook
"""
print(f"  ✅ Ground Truth: {gt_path}")

# Load CSV
df = pd.read_csv(gt_path)
print(f"  📊 Total images: {len(df)}")
print(f"  📊 Columns: {list(df.columns)}")

# ── Convert one-hot to single label ──
# ISIC 2019 uses one-hot encoding: each class is a column with 0/1
label_columns = [col for col in df.columns if col.upper() in ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC', 'UNK']]

if label_columns:
    print(f"  📊 Label columns found: {label_columns}")
    # Get the class with max value (1) for each row
    df['dx'] = df[label_columns].idxmax(axis=1).str.upper()
else:
    # Alternative: single label column
    possible_label_cols = ['diagnosis', 'dx', 'label', 'class']
    for col in possible_label_cols:
        if col in df.columns:
            df['dx'] = df[col].str.upper()
            break

print(f"  📊 Original class distribution:")
print(df['dx'].value_counts().to_string())

# ── Filter: keep only our 5 classes ──
df = df[df['dx'].isin(config.KEEP_CLASSES)].copy()
df['label'] = df['dx'].map(config.ISIC_TO_IDX)
print(f"\n  📊 After filtering (5 classes): {len(df)}")
print(f"  📊 Selected class distribution:")
for dx in config.KEEP_CLASSES:
    count = (df['dx'] == dx).sum()
    idx = config.ISIC_TO_IDX[dx]
    print(f"     [{idx}] {config.CLASS_NAMES[idx]:35s} ({dx:4s}): {count:6d}")

# ── Find image files ──
# ISIC 2019 images are typically in a folder like ISIC_2019_Training_Input/
image_name_col = 'image' if 'image' in df.columns else df.columns[0]
print(f"\n  📊 Image name column: '{image_name_col}'")

# ── แสดงโครงสร้างโฟลเดอร์เพื่อ debug ──
import glob as glob_module
print(f"\n  🔍 Scanning /kaggle/input/ structure...")
for item in sorted(Path("/kaggle/input").rglob("*")):
    # แสดงแค่ 2 ระดับ + นับไฟล์
    depth = len(item.relative_to("/kaggle/input").parts)
    if depth <= 2:
        if item.is_dir():
            file_count = sum(1 for _ in item.glob("*") if _.is_file())
            print(f"     {'  ' * (depth-1)}📂 {item.name}/ ({file_count} files)")
        elif depth <= 1:
            print(f"     📄 {item.name}")

# ── ค้นหารูปภาพแบบ recursive ทั้งหมด ──
print(f"\n  🔍 Searching for images recursively...")
image_paths = {}

# วิธีที่ 1: ค้นหาผ่าน glob recursive ทั้ง /kaggle/input/
for ext in ['jpg', 'jpeg', 'png']:
    for img_file in Path("/kaggle/input").rglob(f"*.{ext}"):
        image_paths[img_file.stem] = str(img_file)
    for img_file in Path("/kaggle/input").rglob(f"*.{ext.upper()}"):
        image_paths[img_file.stem] = str(img_file)

print(f"  📊 Found {len(image_paths)} image files total")

# แสดงตัวอย่าง
if image_paths:
    sample_keys = list(image_paths.keys())[:5]
    print(f"  📊 Sample images: {sample_keys}")
    sample_path = image_paths[sample_keys[0]]
    print(f"  📊 Sample path: {sample_path}")

# Add file path to dataframe  
df['filepath'] = df[image_name_col].map(image_paths)

# If not matched, try with just the filename without ISIC_ prefix variations
matched = df['filepath'].notna().sum()
print(f"  📊 Direct match: {matched}/{len(df)}")

if matched < len(df) * 0.5:
    print(f"  ⚠️ Low match rate, trying alternative matching...")
    for idx, row in df.iterrows():
        if pd.notna(df.at[idx, 'filepath']):
            continue
        name = str(row[image_name_col])
        # Try variations
        for variant in [name, f"ISIC_{name}", name.replace("ISIC_", "")]:
            if variant in image_paths:
                df.at[idx, 'filepath'] = image_paths[variant]
                break

df = df.dropna(subset=['filepath'])
print(f"  📊 Matched images: {len(df)}")

assert len(df) > 100, f"❌ Only matched {len(df)} images! Check dataset path."

# ── Train/Val split (stratified) ──
train_df, val_df = train_test_split(
    df, test_size=0.15, random_state=config.SEED, stratify=df['label']
)
print(f"\n  📊 Train: {len(train_df)} | Val: {len(val_df)}")

# Show per-class split
print(f"\n  📊 Per-class split:")
for dx in config.KEEP_CLASSES:
    idx = config.ISIC_TO_IDX[dx]
    train_count = (train_df['dx'] == dx).sum()
    val_count = (val_df['dx'] == dx).sum()
    print(f"     [{idx}] {config.CLASS_NAMES[idx]:35s}: Train={train_count:5d} | Val={val_count:4d}")
print()


# ═══════════════════════════════════════════════════
#  📦 CELL 4: Dataset & Transforms
# ═══════════════════════════════════════════════════

# DINOv2 normalization (ImageNet stats)
DINOV2_MEAN = [0.485, 0.456, 0.406]
DINOV2_STD = [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
    transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=DINOV2_MEAN, std=DINOV2_STD),
])

val_transforms = transforms.Compose([
    transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=DINOV2_MEAN, std=DINOV2_STD),
])


class SkinCancerDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row['filepath']).convert('RGB')
        label = int(row['label'])
        
        if self.transform:
            image = self.transform(image)
        
        return {"pixel_values": image, "labels": label}


train_dataset = SkinCancerDataset(train_df, transform=train_transforms)
val_dataset = SkinCancerDataset(val_df, transform=val_transforms)

print(f"  ✅ Train dataset: {len(train_dataset)} samples")
print(f"  ✅ Val dataset: {len(val_dataset)} samples")

# Compute class weights for imbalanced data
if config.USE_CLASS_WEIGHTS:
    class_counts = train_df['label'].value_counts().sort_index().values
    total = class_counts.sum()
    class_weights = total / (config.NUM_CLASSES * class_counts)
    class_weights = torch.FloatTensor(class_weights).to(device)
    print(f"\n  ⚖️ Class weights (inverse frequency):")
    for i, (name, w) in enumerate(zip(config.CLASS_NAMES, class_weights)):
        print(f"     [{i}] {name:35s}: {w:.3f}")
else:
    class_weights = None

print()


# ═══════════════════════════════════════════════════
#  📦 CELL 5: Build DINOv2 Model
# ═══════════════════════════════════════════════════

from transformers import Dinov2Model, Dinov2PreTrainedModel, Dinov2Config
from transformers.modeling_outputs import ImageClassifierOutput
import torch.nn as nn

# Fix: transformers ใหม่ต้องการ __main__.__file__ ซึ่งไม่มีใน Notebook
import sys
if not hasattr(sys.modules['__main__'], '__file__') or not sys.modules['__main__'].__file__:
    import tempfile
    _dummy = tempfile.NamedTemporaryFile(suffix='.py', delete=False, mode='w')
    _dummy.write('# dummy file for notebook compatibility\n')
    _dummy.close()
    sys.modules['__main__'].__file__ = _dummy.name


class Dinov2ForSkinCancer(Dinov2PreTrainedModel):
    """
    DINOv2 with classification head for skin cancer detection.
    Uses the [CLS] token output + linear classifier.
    """
    def __init__(self, config):
        super().__init__(config)
        self.dinov2 = Dinov2Model(config)
        self.classifier = nn.Sequential(
            nn.LayerNorm(config.hidden_size),
            nn.Dropout(p=0.1),
            nn.Linear(config.hidden_size, 512),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(512, config.num_labels),
        )
        self.num_labels = config.num_labels
        self.post_init()
    
    def forward(self, pixel_values, labels=None):
        outputs = self.dinov2(pixel_values)
        # Use [CLS] token (first token)
        cls_output = outputs.last_hidden_state[:, 0]
        logits = self.classifier(cls_output)
        
        loss = None
        if labels is not None:
            if class_weights is not None:
                loss_fn = nn.CrossEntropyLoss(weight=class_weights)
            else:
                loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
        
        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


print("🧠 Loading DINOv2 model...")

# Load DINOv2 config and modify for our task
dinov2_config = Dinov2Config.from_pretrained(config.MODEL_NAME)
dinov2_config.num_labels = config.NUM_CLASSES
dinov2_config.id2label = {i: name for i, name in enumerate(config.CLASS_NAMES)}
dinov2_config.label2id = {name: i for i, name in enumerate(config.CLASS_NAMES)}

# Create model with pre-trained DINOv2 backbone
model = Dinov2ForSkinCancer(dinov2_config)

# Load pre-trained DINOv2 weights (backbone only)
pretrained = Dinov2Model.from_pretrained(config.MODEL_NAME)
model.dinov2.load_state_dict(pretrained.state_dict())
del pretrained

# Freeze early layers (optional, for faster training)
# Freeze embeddings + first 8 layers, fine-tune last 4 layers + classifier
for param in model.dinov2.embeddings.parameters():
    param.requires_grad = False

for i, layer in enumerate(model.dinov2.encoder.layer):
    if i < 8:  # Freeze first 8 of 12 layers
        for param in layer.parameters():
            param.requires_grad = False

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"  ✅ Model loaded!")
print(f"  📊 Total params: {total_params:,}")
print(f"  📊 Trainable params: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
print(f"  📊 Frozen params: {total_params - trainable_params:,}")

model = model.to(device)
print()


# ═══════════════════════════════════════════════════
#  📦 CELL 6: Training Setup
# ═══════════════════════════════════════════════════

from torch.optim import AdamW
import math

# DataLoaders
train_loader = DataLoader(
    train_dataset, 
    batch_size=config.TRAIN_BATCH_SIZE, 
    shuffle=True, 
    num_workers=2, 
    pin_memory=True,
    drop_last=True,
)

val_loader = DataLoader(
    val_dataset, 
    batch_size=config.EVAL_BATCH_SIZE, 
    shuffle=False, 
    num_workers=2, 
    pin_memory=True,
)

# Optimizer: AdamW
optimizer = AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=config.LEARNING_RATE,
    weight_decay=config.WEIGHT_DECAY,
    betas=(0.9, 0.999),
    eps=1e-8,
)

# Cosine LR Scheduler with warmup
total_steps = len(train_loader) * config.NUM_EPOCHS
warmup_steps = int(total_steps * config.WARMUP_RATIO)

def lr_lambda(step):
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1 + math.cos(math.pi * progress))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# Mixed precision — ปิดไว้ เพราะ Kaggle P100 (sm_60) มีปัญหา CUDA kernel กับ PyTorch ใหม่
# ถ้าต้องการเปิด AMP ให้ uncomment บรรทัดข้างล่าง (ใช้ได้กับ T4/V100/A100)
# scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
scaler = None  # ปิด AMP — ปลอดภัยกว่า ช้าลงนิดเดียว

print(f"  📊 Total training steps: {total_steps}")
print(f"  📊 Warmup steps: {warmup_steps}")
print(f"  📊 Steps per epoch: {len(train_loader)}")
print()


# ═══════════════════════════════════════════════════
#  📦 CELL 7: Training Loop
# ═══════════════════════════════════════════════════

from sklearn.metrics import f1_score, accuracy_score

def train_one_epoch(model, loader, optimizer, scheduler, scaler, device, epoch):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch_idx, batch in enumerate(loader):
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        scheduler.step()
        
        total_loss += loss.item()
        preds = outputs.logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        
        if (batch_idx + 1) % 10 == 0:
            lr = scheduler.get_last_lr()[0]
            print(f"    Step {batch_idx+1}/{len(loader)} | Loss: {loss.item():.4f} | LR: {lr:.2e}")
    
    avg_loss = total_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    
    return avg_loss, acc, f1


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch in loader:
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)
        
        outputs = model(pixel_values=pixel_values, labels=labels)
        
        total_loss += outputs.loss.item()
        preds = outputs.logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    
    return avg_loss, acc, f1, np.array(all_preds), np.array(all_labels)


# ── Start Training ──
print("=" * 65)
print("  🚀 STARTING TRAINING — DINOv2 + ISIC 2019 (5 classes + SCC)")
print("=" * 65)

best_f1 = 0.0
best_epoch = 0
history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "train_f1": [], "val_f1": []}

for epoch in range(config.NUM_EPOCHS):
    print(f"\n{'─' * 65}")
    print(f"  📊 Epoch {epoch+1}/{config.NUM_EPOCHS}")
    print(f"{'─' * 65}")
    
    # Train
    train_loss, train_acc, train_f1 = train_one_epoch(
        model, train_loader, optimizer, scheduler, scaler, device, epoch
    )
    
    # Evaluate 
    val_loss, val_acc, val_f1, val_preds, val_labels = evaluate(model, val_loader, device)
    
    # Log
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)
    history["train_f1"].append(train_f1)
    history["val_f1"].append(val_f1)
    
    print(f"\n  📈 Results:")
    print(f"     Train — Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | F1: {train_f1:.4f}")
    print(f"     Val   — Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}")
    
    # Save best model
    if val_f1 > best_f1:
        best_f1 = val_f1
        best_epoch = epoch + 1
        
        # Save model
        config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(config.OUTPUT_DIR)
        print(f"     💾 Best model saved! (F1: {best_f1:.4f})")
    
    print(f"     🏆 Best: Epoch {best_epoch} (F1: {best_f1:.4f})")

print(f"\n{'=' * 65}")
print(f"  ✅ TRAINING COMPLETE!")
print(f"  🏆 Best Epoch: {best_epoch} | Best F1: {best_f1:.4f}")
print(f"{'=' * 65}")


# ═══════════════════════════════════════════════════
#  📦 CELL 8: Final Evaluation & Classification Report
# ═══════════════════════════════════════════════════

print("\n📋 Loading best model for final evaluation...")

# Load best model
best_model = Dinov2ForSkinCancer.from_pretrained(config.OUTPUT_DIR)
best_model = best_model.to(device)

# Final evaluation
val_loss, val_acc, val_f1, val_preds, val_labels = evaluate(best_model, val_loader, device)

print(f"\n{'=' * 65}")
print(f"  📊 FINAL CLASSIFICATION REPORT")
print(f"{'=' * 65}")
print(classification_report(
    val_labels, val_preds, 
    target_names=config.CLASS_NAMES,
    digits=4
))

# Confusion Matrix
print(f"\n  📊 Confusion Matrix:")
cm = confusion_matrix(val_labels, val_preds)
print(pd.DataFrame(
    cm, 
    index=[f"True:{n}" for n in config.CLASS_NAMES],
    columns=[f"Pred:{n}" for n in config.CLASS_NAMES]
))


# ═══════════════════════════════════════════════════
#  📦 CELL 9: Save for Production
# ═══════════════════════════════════════════════════

print(f"\n{'=' * 65}")
print(f"  💾 SAVING FOR PRODUCTION")
print(f"{'=' * 65}")

import json

production_config = {
    "architectures": ["Dinov2ForImageClassification"],
    "_name_or_path": "facebook/dinov2-base",
    "model_type": "dinov2",
    "num_labels": config.NUM_CLASSES,
    "id2label": {str(i): name for i, name in enumerate(config.CLASS_NAMES)},
    "label2id": {name: i for i, name in enumerate(config.CLASS_NAMES)},
    "image_size": config.IMAGE_SIZE,
    "hidden_size": 768,
    "num_attention_heads": 12,
    "num_hidden_layers": 12,
    "patch_size": 14,
    "torch_dtype": "float32",
    "training_info": {
        "dataset": "ISIC 2019 (25,331 images)",
        "num_classes": config.NUM_CLASSES,
        "classes": config.CLASS_NAMES,
        "epochs": config.NUM_EPOCHS,
        "learning_rate": config.LEARNING_RATE,
        "batch_size": config.TRAIN_BATCH_SIZE,
        "best_epoch": best_epoch,
        "best_val_f1": float(best_f1),
        "best_val_acc": float(val_acc),
        "optimizer": "AdamW (betas=0.9,0.999, eps=1e-8)",
        "lr_scheduler": "cosine_with_warmup (ratio=0.1)",
        "augmentation": "flip+rotate+colorjitter+affine",
        "class_weights": "inverse_frequency",
        "frozen_layers": "embeddings + first 8/12 transformer layers",
        "mixed_precision": "Native AMP (fp16)",
    }
}

config_path = config.OUTPUT_DIR / "training_config.json"
with open(config_path, 'w') as f:
    json.dump(production_config, f, indent=2, ensure_ascii=False)

print(f"  ✅ Model saved to: {config.OUTPUT_DIR}")
print(f"  📁 Files:")
for f in sorted(config.OUTPUT_DIR.iterdir()):
    size_mb = f.stat().st_size / 1e6
    print(f"     {f.name:30s} ({size_mb:.1f} MB)")

print(f"\n  📋 Class mapping (ใช้ในโปรเจค):")
for i, name in enumerate(config.CLASS_NAMES):
    print(f"     [{i}] {name}")

print(f"\n{'=' * 65}")
print(f"  ✅ ALL DONE! Download model files from: {config.OUTPUT_DIR}")
print(f"  📁 Copy to your project's model folder:")
print(f"     - model.safetensors")
print(f"     - config.json")
print(f"{'=' * 65}")


# ═══════════════════════════════════════════════════
#  📦 CELL 10: Plot Training History (Optional)
# ═══════════════════════════════════════════════════

try:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss
    axes[0].plot(history['train_loss'], label='Train', color='#00647c', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val', color='#007f9d', linewidth=2, linestyle='--')
    axes[0].set_title('Loss', fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Accuracy
    axes[1].plot(history['train_acc'], label='Train', color='#00647c', linewidth=2)
    axes[1].plot(history['val_acc'], label='Val', color='#007f9d', linewidth=2, linestyle='--')
    axes[1].set_title('Accuracy', fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    # F1 Score
    axes[2].plot(history['train_f1'], label='Train', color='#00647c', linewidth=2)
    axes[2].plot(history['val_f1'], label='Val', color='#007f9d', linewidth=2, linestyle='--')
    axes[2].set_title('F1 Score (Macro)', fontweight='bold')
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    plt.suptitle('DINOv2 Skin Cancer (5 Classes + SCC) — Training History', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(config.OUTPUT_DIR / 'training_history.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("  📊 Training plot saved!")
except ImportError:
    print("  ⚠️ matplotlib not available, skipping plots")
