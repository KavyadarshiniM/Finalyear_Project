# ============================================================
# 1. Install required libraries
# ============================================================
!pip install -q segmentation-models-pytorch==0.3.3 nibabel opencv-python-headless scikit-learn tqdm pandas
print("Libraries installed.")


# ============================================================
# 2. Imports
# ============================================================
import os
import glob
import zipfile
import random
import time
import warnings

import numpy as np
import pandas as pd
import nibabel as nib
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")
print("Imports done.")
print("Torch version:", torch.__version__)
print("SMP version:", smp.__version__)


# ============================================================
# 3. Reproducibility - set random seeds
# ============================================================
SEED = 42

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)
print(f"Random seed set to {SEED}")


# ============================================================
# 4. Configuration (edit these if needed)
# ============================================================
CONFIG = {
    "IMG_SIZE": 128,
    "NUM_CLASSES": 4,           # background, necrotic core, edema, enhancing tumor
    "BATCH_SIZE": 4,            # configurable batch size
    "EPOCHS": 40,               # configurable number of epochs
    "LEARNING_RATE": 1e-4,
    "WEIGHT_DECAY": 1e-5,
    "EDGE_SLICES_SKIP": 30,     # skip the first & last N axial slices per patient volume
    "TRAIN_RATIO": 0.9,
    "TEST_RATIO": 0.1,
    "NUM_WORKERS": 2,
    "CHECKPOINT_PATH": "/content/best_model.pth",
    "HISTORY_CSV_PATH": "/content/training_history.csv",
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)
if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
print("Configuration:")
for k, v in CONFIG.items():
    print(f"  {k}: {v}")


# ============================================================
# 5. Mount Google Drive
# ============================================================
from google.colab import drive
drive.mount('/content/drive')


# ============================================================
# 6. Provide ZIP path & extract dataset
# ============================================================
ZIP_PATH = "/content/drive/MyDrive/BraTS2020_TrainingData.zip"  # hardcoded path

if not os.path.isfile(ZIP_PATH):
    raise FileNotFoundError(f"Could not find zip file at: {ZIP_PATH}\nPlease check the path and try again.")

EXTRACT_DIR = "/content/BraTS2020_data"
os.makedirs(EXTRACT_DIR, exist_ok=True)

# Only extract if not already extracted (saves time on re-runs)
already_extracted = False
for root, dirs, files in os.walk(EXTRACT_DIR):
    if any(d.startswith("BraTS20_Training") for d in dirs):
        already_extracted = True
        break

if already_extracted:
    print("Dataset already extracted, skipping extraction.")
else:
    print("Extracting zip file... this may take a few minutes.")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(EXTRACT_DIR)
    print("Extraction complete.")


# ============================================================
# 7. Auto-detect dataset root and patient folders
# ============================================================
# find_data_root: walks the extracted folder tree and finds the directory
# that directly contains the patient folders (BraTS20_Training_XXX).
def find_data_root(base_dir):
    for root, dirs, files in os.walk(base_dir):
        if any(d.startswith("BraTS20_Training") for d in dirs):
            return root
    return None

DATA_ROOT = find_data_root(EXTRACT_DIR)
if DATA_ROOT is None:
    raise FileNotFoundError(
        "Could not locate 'BraTS20_Training_XXX' patient folders inside the extracted zip. "
        "Please check that the zip file has the expected BraTS2020 structure."
    )
print("Detected dataset root:", DATA_ROOT)

REQUIRED_SUFFIXES = ["_flair.nii", "_t1.nii", "_t1ce.nii", "_t2.nii", "_seg.nii"]

# get_valid_patient_dirs: returns patient directories that contain all 5
# required modality files, and reports any patients that are missing files.
def get_valid_patient_dirs(data_root):
    all_dirs = sorted(glob.glob(os.path.join(data_root, "BraTS20_Training_*")))
    valid_dirs = []
    skipped = []
    for d in all_dirs:
        pid = os.path.basename(d)
        missing = [s for s in REQUIRED_SUFFIXES if not os.path.isfile(os.path.join(d, pid + s))]
        if missing:
            skipped.append((pid, missing))
        else:
            valid_dirs.append(d)
    return valid_dirs, skipped

patient_dirs, skipped_patients = get_valid_patient_dirs(DATA_ROOT)

print(f"Found {len(patient_dirs)} valid patient folders.")
if skipped_patients:
    print(f"Skipped {len(skipped_patients)} patient folders due to missing files, e.g.:")
    for pid, missing in skipped_patients[:5]:
        print(f"  {pid}: missing {missing}")

if len(patient_dirs) == 0:
    raise RuntimeError("No valid patient folders found. Cannot continue.")


# ============================================================
# 8. Build a slice-level index (2D axial slices, skipping edge slices)
# ============================================================
# Instead of filtering slices by tumor-pixel count, we simply skip the
# first and last EDGE_SLICES_SKIP axial slices of each volume (these
# tend to contain little to no brain/tumor tissue) and keep every slice
# in between. Only the segmentation volume's shape is read here (no
# voxel data is loaded), so this is fast.
def build_slice_index(patient_dirs, edge_skip=30):
    samples = []
    for d in tqdm(patient_dirs, desc="Indexing slices"):
        pid = os.path.basename(d)
        seg_path = os.path.join(d, pid + "_seg.nii")
        try:
            n_slices = nib.load(seg_path).shape[2]
        except Exception as e:
            print(f"Warning: could not read {seg_path} ({e}). Skipping patient.")
            continue
        start = edge_skip
        end = n_slices - edge_skip
        if end <= start:
            continue  # volume too short after skipping edge slices
        for s in range(start, end):
            samples.append((d, s))
    return samples

all_samples = build_slice_index(patient_dirs, CONFIG["EDGE_SLICES_SKIP"])
print(f"Total usable 2D slices across all patients: {len(all_samples)}")


# ============================================================
# 9. Split PATIENTS (not slices) into train / test
# ============================================================
train_patients, test_patients = train_test_split(
    patient_dirs, train_size=CONFIG["TRAIN_RATIO"], random_state=SEED
)

print(f"Patients -> train: {len(train_patients)}, test: {len(test_patients)}")

train_patient_set = set(train_patients)
test_patient_set = set(test_patients)

train_samples = [s for s in all_samples if s[0] in train_patient_set]
test_samples = [s for s in all_samples if s[0] in test_patient_set]

print(f"Slices  -> train: {len(train_samples)}, test: {len(test_samples)}")


# ============================================================
# 10. Dataset class - loads 4-channel MRI slices + segmentation mask
# ============================================================
MODALITY_SUFFIXES = {
    "flair": "_flair.nii",
    "t1": "_t1.nii",
    "t1ce": "_t1ce.nii",
    "t2": "_t2.nii",
}

# normalize_slice: z-score normalizes using nonzero (brain) voxels,
# then rescales the result to the [0, 1] range.
def normalize_slice(slice_2d):
    slice_2d = slice_2d.astype(np.float32)
    mask = slice_2d > 0
    if mask.sum() > 0:
        mean = slice_2d[mask].mean()
        std = slice_2d[mask].std() + 1e-8
        slice_2d = (slice_2d - mean) / std
        slice_2d = np.clip(slice_2d, -5, 5)
    # rescale to [0, 1]
    min_v, max_v = slice_2d.min(), slice_2d.max()
    if max_v - min_v > 1e-8:
        slice_2d = (slice_2d - min_v) / (max_v - min_v)
    else:
        slice_2d = np.zeros_like(slice_2d)
    return slice_2d


# BraTSDataset: loads individual 2D axial slices on demand from the 3D
# .nii volumes, using nibabel's lazy dataobj slicing so full volumes are
# never fully loaded into memory at once.
class BraTSDataset(Dataset):

    def __init__(self, samples, img_size=128, augment=False):
        self.samples = samples
        self.img_size = img_size
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def _load_modality_slice(self, patient_dir, pid, suffix, slice_idx):
        path = os.path.join(patient_dir, pid + suffix)
        try:
            img = nib.load(path)
            slice_2d = np.asarray(img.dataobj[:, :, slice_idx])
        except Exception as e:
            raise RuntimeError(f"Failed to read slice {slice_idx} from {path}: {e}")
        return slice_2d

    def __getitem__(self, idx):
        patient_dir, slice_idx = self.samples[idx]
        pid = os.path.basename(patient_dir)

        channels = []
        for key in ["flair", "t1", "t1ce", "t2"]:
            raw_slice = self._load_modality_slice(patient_dir, pid, MODALITY_SUFFIXES[key], slice_idx)
            norm_slice = normalize_slice(raw_slice)
            resized = cv2.resize(norm_slice, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
            channels.append(resized)

        seg_raw = self._load_modality_slice(patient_dir, pid, "_seg.nii", slice_idx)
        seg_resized = cv2.resize(seg_raw.astype(np.float32), (self.img_size, self.img_size),
                                  interpolation=cv2.INTER_NEAREST)

        # BraTS labels are 0, 1, 2, 4 -> remap label 4 to 3 for contiguous class indices
        seg_resized = seg_resized.astype(np.int64)
        seg_resized[seg_resized == 4] = 3

        image = np.stack(channels, axis=0).astype(np.float32)   # shape: (4, H, W)
        mask = seg_resized                                      # shape: (H, W)

        # simple augmentation: random horizontal/vertical flip
        if self.augment:
            if random.random() < 0.5:
                image = np.flip(image, axis=2).copy()
                mask = np.flip(mask, axis=1).copy()
            if random.random() < 0.5:
                image = np.flip(image, axis=1).copy()
                mask = np.flip(mask, axis=0).copy()

        return torch.from_numpy(image), torch.from_numpy(mask)


# ============================================================
# 11. Create PyTorch Datasets & DataLoaders
# ============================================================
train_dataset = BraTSDataset(train_samples, img_size=CONFIG["IMG_SIZE"], augment=True)
test_dataset = BraTSDataset(test_samples, img_size=CONFIG["IMG_SIZE"], augment=False)

train_loader = DataLoader(train_dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=True,
                           num_workers=CONFIG["NUM_WORKERS"], pin_memory=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=False,
                          num_workers=CONFIG["NUM_WORKERS"], pin_memory=True)

print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

# Sanity check: fetch one batch and print shapes
sample_images, sample_masks = next(iter(train_loader))
print("Image batch shape:", sample_images.shape)   # (B, 4, H, W)
print("Mask batch shape:", sample_masks.shape)      # (B, H, W)
print("Unique mask values in batch:", torch.unique(sample_masks))


# ============================================================
# 12. Quick sanity-check visualization of one training sample
# ============================================================
img, msk = train_dataset[0]
flair_channel = img[0].numpy()

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
axes[0].imshow(flair_channel, cmap="gray")
axes[0].set_title("FLAIR slice")
axes[0].axis("off")
axes[1].imshow(msk.numpy(), cmap="jet", vmin=0, vmax=3)
axes[1].set_title("Ground truth mask")
axes[1].axis("off")
plt.tight_layout()
plt.show()


# ============================================================
# 13. Build U-Net++ model (ResNet34 encoder, ImageNet pretrained)
# ============================================================
model = smp.UnetPlusPlus(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=4,              # FLAIR, T1, T1CE, T2
    classes=CONFIG["NUM_CLASSES"],
    activation=None,            # raw logits; softmax/argmax applied where needed
)
model = model.to(DEVICE)

num_params = sum(p.numel() for p in model.parameters())
print(f"Model created: U-Net++ (ResNet34 encoder). Total parameters: {num_params:,}")


# ============================================================
# 14. Loss function: Dice Loss + Cross Entropy Loss
# ============================================================
class DiceCELoss(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.dice = smp.losses.DiceLoss(mode="multiclass", from_logits=True)

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return ce_loss + dice_loss

criterion = DiceCELoss(CONFIG["NUM_CLASSES"])
print("Loss function: CrossEntropyLoss + DiceLoss")


# ============================================================
# 15. Metric helper functions (Accuracy, Dice/F1, IoU, Precision, Recall)
# ============================================================
# BraTS class indices used in this pipeline (after remapping label 4 -> 3):
#   0 = Background, 1 = Necrotic core, 2 = Edema, 3 = Enhancing tumor
#   Tumor Core (BraTS convention) = Necrotic core (1) + Enhancing tumor (3)

def dice_coefficient(preds, targets, class_ids, eps=1e-6):
    """Dice score for a single class or a combined group of classes.
    preds, targets: (B, H, W) integer label tensors.
    class_ids: list of class indices that make up the positive region.
    """
    pred_mask = torch.zeros_like(preds, dtype=torch.bool)
    target_mask = torch.zeros_like(targets, dtype=torch.bool)
    for c in class_ids:
        pred_mask |= (preds == c)
        target_mask |= (targets == c)

    pred_mask = pred_mask.float()
    target_mask = target_mask.float()

    intersection = (pred_mask * target_mask).sum()
    dice = (2.0 * intersection + eps) / (pred_mask.sum() + target_mask.sum() + eps)
    return dice.item()


# Computes accuracy, overall Dice(F1), IoU, precision, recall (micro-averaged
# over all classes) plus the region-specific Dice scores (background / edema /
# enhancing tumor / tumor core) for a batch.
def compute_batch_metrics(logits, targets, num_classes):
    preds = torch.argmax(logits, dim=1)

    tp, fp, fn, tn = smp.metrics.get_stats(
        preds, targets, mode="multiclass", num_classes=num_classes
    )
    accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction="micro").item()
    dice = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro").item()
    iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro").item()
    precision = smp.metrics.precision(tp, fp, fn, tn, reduction="micro").item()
    recall = smp.metrics.recall(tp, fp, fn, tn, reduction="micro").item()

    background_dice = dice_coefficient(preds, targets, class_ids=[0])
    edema_dice = dice_coefficient(preds, targets, class_ids=[2])
    enhancing_dice = dice_coefficient(preds, targets, class_ids=[3])
    tumor_core_dice = dice_coefficient(preds, targets, class_ids=[1, 3])

    return {
        "accuracy": accuracy,
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "background_dice": background_dice,
        "edema_dice": edema_dice,
        "enhancing_dice": enhancing_dice,
        "tumor_core_dice": tumor_core_dice,
    }


# ============================================================
# 16. Optimizer, Scheduler, AMP scaler
# ============================================================
optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["LEARNING_RATE"],
                               weight_decay=CONFIG["WEIGHT_DECAY"])

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=3
)

scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == "cuda"))

print("Optimizer: AdamW | Scheduler: ReduceLROnPlateau | AMP enabled:", DEVICE.type == "cuda")
print(f"Training will run for the full {CONFIG['EPOCHS']} epochs (no early stopping).")


# ============================================================
# 17. Training & Validation loop function
# ============================================================
METRIC_KEYS = [
    "accuracy", "dice", "iou", "precision", "recall",
    "background_dice", "edema_dice", "enhancing_dice", "tumor_core_dice",
]

def run_epoch(model, loader, criterion, optimizer, scaler, device, num_classes, train=True):
    model.train() if train else model.eval()

    total_loss = 0.0
    metric_sums = {k: 0.0 for k in METRIC_KEYS}
    n_batches = 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, masks in tqdm(loader, desc="Train" if train else "Val", leave=False):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            if train:
                optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                logits = model(images)
                loss = criterion(logits, masks)

            if train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            batch_metrics = compute_batch_metrics(logits.detach(), masks, num_classes)
            for k in metric_sums:
                metric_sums[k] += batch_metrics[k]

            total_loss += loss.item()
            n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    avg_metrics = {k: v / max(n_batches, 1) for k, v in metric_sums.items()}
    return avg_loss, avg_metrics


# ============================================================
# 18. Main training loop (with checkpointing)
# ============================================================
history = {
    "epoch": [], "train_loss": [], "train_accuracy": [],
    "train_dice": [], "train_background_dice": [], "train_edema_dice": [],
    "train_enhancing_dice": [], "train_tumor_core_dice": [],
    "lr": [], "epoch_time_sec": [],
}

best_train_dice = -1.0
training_start_time = time.time()

print("Starting training...\n")
for epoch in range(1, CONFIG["EPOCHS"] + 1):
    start_time = time.time()

    train_loss, train_metrics = run_epoch(
        model, train_loader, criterion, optimizer, scaler, DEVICE, CONFIG["NUM_CLASSES"], train=True
    )

    scheduler.step(train_loss)
    current_lr = optimizer.param_groups[0]["lr"]
    epoch_time = time.time() - start_time

    # Save history (for plotting later)
    history["epoch"].append(epoch)
    history["train_loss"].append(train_loss)
    history["train_accuracy"].append(train_metrics["accuracy"])
    history["train_dice"].append(train_metrics["dice"])
    history["train_background_dice"].append(train_metrics["background_dice"])
    history["train_edema_dice"].append(train_metrics["edema_dice"])
    history["train_enhancing_dice"].append(train_metrics["enhancing_dice"])
    history["train_tumor_core_dice"].append(train_metrics["tumor_core_dice"])
    history["lr"].append(current_lr)
    history["epoch_time_sec"].append(epoch_time)

    print(
        f"Epoch [{epoch}/{CONFIG['EPOCHS']}] | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_metrics['accuracy']:.4f} | "
        f"Dice: {train_metrics['dice']:.4f} | "
        f"Background Dice: {train_metrics['background_dice']:.4f} | "
        f"Enhancing Dice: {train_metrics['enhancing_dice']:.4f} | "
        f"Tumor Core Dice: {train_metrics['tumor_core_dice']:.4f} | "
        f"Edema Dice: {train_metrics['edema_dice']:.4f} | "
        f"LR: {current_lr:.2e} | Time: {epoch_time:.1f}s"
    )

    # Model checkpoint - save the best model based on training Dice score
    if train_metrics["dice"] > best_train_dice:
        best_train_dice = train_metrics["dice"]
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_dice": best_train_dice,
            "config": CONFIG,
        }, CONFIG["CHECKPOINT_PATH"])
        print(f"  -> New best model saved (Train Dice: {best_train_dice:.4f})")

total_training_time_sec = time.time() - training_start_time

print("\nTraining complete.")
print(f"Best training Dice score: {best_train_dice:.4f}")
print(f"Total training time: {total_training_time_sec/60:.2f} minutes ({total_training_time_sec:.1f} seconds)")

# Save training history to CSV
history_df = pd.DataFrame(history)
history_df.to_csv(CONFIG["HISTORY_CSV_PATH"], index=False)
print(f"Training history saved to {CONFIG['HISTORY_CSV_PATH']}")


# ============================================================
# 19. Plot training curves
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(history_df["epoch"], history_df["train_loss"], label="Train Loss")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss"); axes[0].set_title("Loss Curve")
axes[0].legend(); axes[0].grid(True)

axes[1].plot(history_df["epoch"], history_df["train_dice"], label="Train Dice")
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Dice Score"); axes[1].set_title("Dice Score Curve")
axes[1].legend(); axes[1].grid(True)

axes[2].plot(history_df["epoch"], history_df["train_accuracy"], label="Train Accuracy")
axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("Accuracy"); axes[2].set_title("Accuracy Curve")
axes[2].legend(); axes[2].grid(True)

plt.tight_layout()
plt.savefig("/content/training_curves.png", dpi=150)
plt.show()


# ============================================================
# 20. Load best model & evaluate on the TEST set
# ============================================================
checkpoint = torch.load(CONFIG["CHECKPOINT_PATH"], map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
print(f"Loaded best model from epoch {checkpoint['epoch']} (Train Dice: {checkpoint['train_dice']:.4f})")

test_loss, test_metrics = run_epoch(
    model, test_loader, criterion, optimizer, scaler, DEVICE, CONFIG["NUM_CLASSES"], train=False
)

# Accumulate stats over the whole test set to also report macro-averaged F1
all_tp, all_fp, all_fn, all_tn = [], [], [], []
with torch.no_grad():
    for images, masks in tqdm(test_loader, desc="Final Test Eval"):
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)
        with torch.cuda.amp.autocast(enabled=(DEVICE.type == "cuda")):
            logits = model(images)
        preds = torch.argmax(logits, dim=1)
        tp, fp, fn, tn = smp.metrics.get_stats(preds, masks, mode="multiclass", num_classes=CONFIG["NUM_CLASSES"])
        all_tp.append(tp); all_fp.append(fp); all_fn.append(fn); all_tn.append(tn)

all_tp = torch.cat(all_tp); all_fp = torch.cat(all_fp)
all_fn = torch.cat(all_fn); all_tn = torch.cat(all_tn)

test_f1_macro = smp.metrics.f1_score(all_tp, all_fp, all_fn, all_tn, reduction="macro").item()

print("\n===== TEST SET RESULTS =====")
print(f"Test Loss:            {test_loss:.4f}")
print(f"Test Accuracy:        {test_metrics['accuracy']:.4f}")
print(f"Dice Score:           {test_metrics['dice']:.4f}")
print(f"Background Dice:      {test_metrics['background_dice']:.4f}")
print(f"Enhancing Tumor Dice: {test_metrics['enhancing_dice']:.4f}")
print(f"Tumor Core Dice:      {test_metrics['tumor_core_dice']:.4f}")
print(f"Edema Dice:           {test_metrics['edema_dice']:.4f}")
print(f"IoU:                  {test_metrics['iou']:.4f}")
print(f"Precision:            {test_metrics['precision']:.4f}")
print(f"Recall:               {test_metrics['recall']:.4f}")
print(f"F1 Score (macro):     {test_f1_macro:.4f}")

# Store test metrics for later use / plotting
test_results = {
    "test_loss": test_loss,
    "test_f1_macro": test_f1_macro,
    **{f"test_{k}": v for k, v in test_metrics.items()},
}


# ============================================================
# 21. Visualize predictions: FLAIR | Ground Truth | Prediction
# ============================================================
def visualize_predictions(model, dataset, device, num_samples=5):
    model.eval()
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))

    fig, axes = plt.subplots(len(indices), 3, figsize=(12, 4 * len(indices)))
    if len(indices) == 1:
        axes = axes.reshape(1, -1)

    with torch.no_grad():
        for row, idx in enumerate(indices):
            image, mask = dataset[idx]
            input_tensor = image.unsqueeze(0).to(device)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                logits = model(input_tensor)
            pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()

            flair_img = image[0].numpy()
            gt_mask = mask.numpy()

            axes[row, 0].imshow(flair_img, cmap="gray")
            axes[row, 0].set_title("FLAIR Image")
            axes[row, 0].axis("off")

            axes[row, 1].imshow(flair_img, cmap="gray")
            axes[row, 1].imshow(gt_mask, cmap="jet", alpha=0.5, vmin=0, vmax=3)
            axes[row, 1].set_title("Ground Truth Mask")
            axes[row, 1].axis("off")

            axes[row, 2].imshow(flair_img, cmap="gray")
            axes[row, 2].imshow(pred, cmap="jet", alpha=0.5, vmin=0, vmax=3)
            axes[row, 2].set_title("Predicted Mask")
            axes[row, 2].axis("off")

    plt.tight_layout()
    plt.savefig("/content/prediction_samples.png", dpi=150)
    plt.show()

visualize_predictions(model, test_dataset, DEVICE, num_samples=5)

print("\nAll done!")
print(f"Best model checkpoint: {CONFIG['CHECKPOINT_PATH']}")
print(f"Training history CSV:  {CONFIG['HISTORY_CSV_PATH']}")


# ============================================================
# 22. Model parameter summary & total training time
# ============================================================
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
non_trainable_params = total_params - trainable_params

print("===== MODEL SUMMARY =====")
print(f"Total parameters:         {total_params:,}")
print(f"Trainable parameters:     {trainable_params:,}")
print(f"Non-trainable parameters: {non_trainable_params:,}")
print(f"\nTotal training time: {total_training_time_sec/60:.2f} minutes ({total_training_time_sec:.1f} seconds)")

