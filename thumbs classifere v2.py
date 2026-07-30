"""
Thumb Angle Regressor
======================
Trains a CNN to output a value from 0 (thumbs down) to 1 (thumbs up)
based on the angle of your thumb, extracted automatically using MediaPipe.

HOW TO USE THIS SCRIPT
-----------------------
1. Install dependencies (run in Thonny's shell, or your terminal):

       pip install opencv-python mediapipe torch torchvision pandas pillow

   (If you're on Windows and torch install fails, go to
   https://pytorch.org/get-started/locally/ and use the install command
   it gives you for your system instead.)

2. Record several short videos of yourself sweeping your thumb from
   fully DOWN to fully UP (and maybe back down), varying background,
   lighting, and distance between videos. Save them all as .mp4 files
   in a single folder (e.g. a folder called "videos").

3. Set VIDEO_DIR below to point to that folder.

4. Run this script. It will, in order:
     a) Go through every video in VIDEO_DIR and extract frames from
        each into ./frames/ (frame filenames are prefixed with the
        video's filename so nothing collides)
     b) Use MediaPipe to detect your hand and compute the thumb angle
        for each new frame, saving labels to ./labels.csv
     c) Train a small CNN regressor on all labeled frames so far
     d) Save the trained model to ./thumb_regressor.pth
     e) Show you a quick plot of predicted vs "true" value on a
        held-out portion of your data

5. Got more videos later? Just drop them into VIDEO_DIR and re-run --
   already-processed videos and frames are automatically skipped, so
   only the new videos get extracted and labeled, then the model
   retrains on the full accumulated dataset.

NOTE ON THONNY
---------------
Thonny runs plain Python scripts fine. Just open this file in Thonny
and hit Run (F5). Training may take a few minutes depending on your
PC and how many frames you have -- this is normal, just let it run.

IMPORTANT -- IF YOU'VE RUN AN OLDER VERSION OF THIS SCRIPT BEFORE:
--------------------------------------------------------------------
This version crops frames to just the hand region before training
(instead of using the full frame), which needs a fresh labels.csv to
regenerate crops for every frame. Before running this version, delete
(or rename) your existing labels.csv file -- otherwise old frames will
be skipped as "already labeled" and never get cropped. Your raw
videos/ and frames/ folders are unaffected and don't need to be
redone, only labels.csv needs to go.
"""

import os
import csv
import math
import glob

# ============================================================
# CONFIG -- edit these before running
# ============================================================

VIDEO_DIR = "videos"      # <-- folder containing all your .mp4 sweep videos

FRAME_DIR = "frames"              # raw extracted frames
CROPPED_FRAME_DIR = "frames_cropped"  # hand-cropped frames, used for training
CROP_MARGIN = 0.4                 # extra padding around detected hand box (fraction of box size)
LABELS_CSV = "labels.csv"
MODEL_PATH = "thumb_regressor.pth"

FPS_EXTRACT = 10          # how many frames per second to pull from the video
EPOCHS = 20
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
IMG_SIZE = 224

# Set to False to skip frame extraction / labeling if you've already
# done it and just want to re-train (e.g. after adding more videos).
DO_EXTRACT_FRAMES = True
DO_LABEL_FRAMES = True
DO_TRAIN = True


# ============================================================
# STEP 1: Extract frames from every video in a folder using OpenCV
# ============================================================

PROCESSED_VIDEOS_LOG = "processed_videos.txt"


def _load_processed_videos(log_path):
    if not os.path.exists(log_path):
        return set()
    with open(log_path, "r") as f:
        return set(line.strip() for line in f if line.strip())


def _mark_video_processed(log_path, video_filename):
    with open(log_path, "a") as f:
        f.write(video_filename + "\n")


def extract_frames_from_video(video_path, out_dir, video_name, target_fps=10):
    import cv2

    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if src_fps <= 0:
        src_fps = 30  # fallback assumption
    frame_interval = max(1, round(src_fps / target_fps))

    count = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_interval == 0:
            fname = f"{video_name}_{saved:04d}.jpg"
            cv2.imwrite(os.path.join(out_dir, fname), frame)
            saved += 1
        count += 1
    cap.release()
    print(f"[extract_frames] Saved {saved} frames from {video_path} to {out_dir}/")


def extract_frames_from_folder(video_dir, out_dir, target_fps=10):
    if not os.path.isdir(video_dir):
        raise RuntimeError(
            f"Video folder '{video_dir}' not found. Create it and put your "
            f".mp4 files inside, or update VIDEO_DIR at the top of the script."
        )

    video_extensions = (".mp4", ".mov", ".avi", ".mkv")
    video_files = sorted(
        f for f in os.listdir(video_dir) if f.lower().endswith(video_extensions)
    )

    if not video_files:
        raise RuntimeError(f"No video files found in '{video_dir}'.")

    processed = _load_processed_videos(PROCESSED_VIDEOS_LOG)
    new_count = 0

    for video_file in video_files:
        if video_file in processed:
            print(f"[extract_frames] Skipping already-processed video: {video_file}")
            continue

        video_path = os.path.join(video_dir, video_file)
        video_name = os.path.splitext(video_file)[0]  # e.g. "take1" from "take1.mp4"
        extract_frames_from_video(video_path, out_dir, video_name, target_fps=target_fps)
        _mark_video_processed(PROCESSED_VIDEOS_LOG, video_file)
        new_count += 1

    print(f"[extract_frames] Processed {new_count} new video(s) out of {len(video_files)} found.")


# ============================================================
# STEP 2: Label frames using MediaPipe hand landmarks
# ============================================================

def angle_to_label(angle_deg):
    """Map thumb angle in degrees (-90=down, 90=up) to a 0-1 label."""
    clipped = max(-90.0, min(90.0, angle_deg))
    return (clipped + 90.0) / 180.0


def crop_hand_region(img, landmarks, margin=0.4):
    """
    Given an image (numpy array, BGR or RGB, doesn't matter for cropping)
    and MediaPipe hand landmarks, return a crop tightly around the hand
    with extra padding (margin) on each side. Returns None if the
    resulting crop would be empty/invalid.
    """
    h, w = img.shape[:2]
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    box_w = x_max - x_min
    box_h = y_max - y_min

    # pad on each side by margin * box size (with a small floor so tiny
    # boxes still get reasonable padding)
    pad_w = max(box_w, 0.05) * margin
    pad_h = max(box_h, 0.05) * margin

    x_min -= pad_w
    x_max += pad_w
    y_min -= pad_h
    y_max += pad_h

    # clip to valid [0, 1] range
    x_min = max(0.0, x_min)
    x_max = min(1.0, x_max)
    y_min = max(0.0, y_min)
    y_max = min(1.0, y_max)

    px_min, px_max = int(x_min * w), int(x_max * w)
    py_min, py_max = int(y_min * h), int(y_max * h)

    if px_max <= px_min or py_max <= py_min:
        return None

    return img[py_min:py_max, px_min:px_max]


def label_frames(frame_dir, cropped_dir, csv_path, crop_margin=0.4):
    import cv2
    import mediapipe as mp
    import mediapipe.python.solutions.hands as mp_hands_module  # noqa: F401 -- forces solutions to load on some mediapipe builds
    import numpy as np

    os.makedirs(cropped_dir, exist_ok=True)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5,
    )

    # Load existing labels if the CSV already exists, so we can append
    # new frames from a different video without losing old ones.
    existing_rows = []
    existing_files = set()
    if os.path.exists(csv_path):
        with open(csv_path, "r", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if row:
                    existing_rows.append(row)
                    existing_files.add(row[0])

    new_rows = []
    all_files = sorted(os.listdir(frame_dir))
    skipped_no_hand = 0
    skipped_bad_crop = 0

    for fname in all_files:
        if fname in existing_files:
            continue  # already labeled (and cropped) in a previous run
        path = os.path.join(frame_dir, fname)
        img = cv2.imread(path)
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if not results.multi_hand_landmarks:
            skipped_no_hand += 1
            continue

        lm = results.multi_hand_landmarks[0].landmark

        tip = np.array([lm[4].x, lm[4].y])   # thumb tip
        base = np.array([lm[2].x, lm[2].y])  # thumb base (CMC joint)
        vec = tip - base

        # image y-axis grows downward, flip so "up" is positive
        angle_rad = math.atan2(-vec[1], vec[0])
        angle_deg = math.degrees(angle_rad)
        label = angle_to_label(angle_deg)

        # crop tightly around the whole hand (all 21 landmarks) with padding
        crop = crop_hand_region(img, lm, margin=crop_margin)
        if crop is None:
            skipped_bad_crop += 1
            continue

        cv2.imwrite(os.path.join(cropped_dir, fname), crop)
        new_rows.append([fname, f"{label:.4f}"])

    hands.close()

    all_rows = existing_rows + new_rows
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label"])
        writer.writerows(all_rows)

    print(f"[label_frames] Labeled + cropped {len(new_rows)} new frames "
          f"({skipped_no_hand} skipped: no hand detected, "
          f"{skipped_bad_crop} skipped: bad crop).")
    print(f"[label_frames] Total labeled frames in {csv_path}: {len(all_rows)}")


# ============================================================
# STEP 3: Dataset + Model + Training
# ============================================================

def train_model(csv_path, frame_dir, model_path):
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader, random_split
    from torchvision import models, transforms
    from PIL import Image
    import pandas as pd

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train_model] Using device: {device}")

    # ---- Dataset ----
    class ThumbDataset(Dataset):
        def __init__(self, dataframe, img_dir, transform=None):
            self.data = dataframe.reset_index(drop=True)
            self.img_dir = img_dir
            self.transform = transform

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            row = self.data.iloc[idx]
            img = Image.open(os.path.join(self.img_dir, row["filename"])).convert("RGB")
            label = torch.tensor(float(row["label"]), dtype=torch.float32)
            if self.transform:
                img = self.transform(img)
            return img, label

    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    df = pd.read_csv(csv_path)
    if len(df) < 20:
        raise RuntimeError(
            f"Only {len(df)} labeled frames found -- that's not enough to train on. "
            "Record a longer/slower sweep video, or add more takes."
        )

    # Simple random split (fine for a first pass; for best practice, split
    # by video/session instead so validation isn't near-duplicate frames)
    full_dataset = ThumbDataset(df, frame_dir, transform=train_transform)
    val_dataset_full = ThumbDataset(df, frame_dir, transform=val_transform)

    train_size = int(0.85 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    generator = torch.Generator().manual_seed(42)
    train_indices, val_indices = random_split(
        range(len(full_dataset)), [train_size, val_size], generator=generator
    )

    train_ds = torch.utils.data.Subset(full_dataset, train_indices.indices)
    val_ds = torch.utils.data.Subset(val_dataset_full, val_indices.indices)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # ---- Model ----
    class ThumbRegressor(nn.Module):
        def __init__(self):
            super().__init__()
            backbone = models.mobilenet_v3_small(weights="IMAGENET1K_V1")
            backbone.classifier[-1] = nn.Linear(backbone.classifier[-1].in_features, 1)
            self.backbone = backbone

        def forward(self, x):
            return torch.sigmoid(self.backbone(x)).squeeze(1)

    model = ThumbRegressor().to(device)
    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float("inf")

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            preds = model(imgs)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                preds = model(imgs)
                val_loss += criterion(preds, labels).item() * imgs.size(0)
        val_loss /= max(1, len(val_ds))

        print(f"Epoch {epoch+1}/{EPOCHS} - train_loss: {train_loss:.4f} - val_loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)

    print(f"[train_model] Done. Best val loss: {best_val_loss:.4f}. Model saved to {model_path}")

    # ---- Quick sanity check plot (predicted vs true on val set) ----
    try:
        import matplotlib.pyplot as plt

        model.eval()
        all_preds, all_true = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                preds = model(imgs).cpu().numpy()
                all_preds.extend(preds.tolist())
                all_true.extend(labels.numpy().tolist())

        plt.figure(figsize=(5, 5))
        plt.scatter(all_true, all_preds, alpha=0.6)
        plt.plot([0, 1], [0, 1], "r--", label="perfect prediction")
        plt.xlabel("True label (angle-based)")
        plt.ylabel("Predicted value")
        plt.title("Validation: predicted vs true")
        plt.legend()
        plt.tight_layout()
        plt.savefig("validation_plot.png")
        print("[train_model] Saved validation_plot.png -- open it to check predictions.")
    except ImportError:
        print("[train_model] matplotlib not installed, skipping plot (pip install matplotlib to enable).")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    if DO_EXTRACT_FRAMES:
        extract_frames_from_folder(VIDEO_DIR, FRAME_DIR, target_fps=FPS_EXTRACT)

    if DO_LABEL_FRAMES:
        label_frames(FRAME_DIR, CROPPED_FRAME_DIR, LABELS_CSV, crop_margin=CROP_MARGIN)

    if DO_TRAIN:
        train_model(LABELS_CSV, CROPPED_FRAME_DIR, MODEL_PATH)

    print("\nAll done! To classify new images/webcam frames, load the model with:")
    print('  model.load_state_dict(torch.load("thumb_regressor.pth"))')
    print("and run it on a new image the same way as val_transform above.")

