"""
thumb_control.py
==================
Reusable background webcam + MediaPipe + thumb-angle-regressor module.
Runs hand detection and model inference in a separate thread so games
can run their own loop at full framerate without waiting on the
camera/model each frame.

SUPPORTS UP TO TWO HANDS, classified by screen position:
    - a hand on the left side of the frame -> "left"
    - a hand on the right side of the frame -> "right"
Each side gets its own independently smoothed 0.0-1.0 value, so two
people (or one person using both hands) can each control something
different at the same time.

USAGE (single-hand games, e.g. the current Pong):

    from thumb_control import ThumbController

    controller = ThumbController()
    controller.start()

    value = controller.get_value()             # whichever hand is seen
    detected = controller.get_hand_detected()   # True if any hand seen

    controller.stop()

USAGE (two-hand / two-player games):

    left_value = controller.get_left_value()
    right_value = controller.get_right_value()
    left_seen = controller.get_left_detected()
    right_seen = controller.get_right_detected()

    boxes = controller.get_hand_boxes()   # {"left": (x1,y1,x2,y2) or None,
                                           #  "right": (x1,y1,x2,y2) or None}
                                           # pixel coords in the raw camera frame,
                                           # handy for drawing overlays

Requires thumb_regressor.pth (from thumb_regressor.py training) to be
in the same folder.
"""

import threading
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import mediapipe as mp
import mediapipe.python.solutions.hands as mp_hands_module  # noqa: F401 -- forces solutions to load on some mediapipe builds

# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = "thumb_regressor.pth"
IMG_SIZE = 224
CAMERA_INDEX = 0
CROP_MARGIN = 0.4      # must match CROP_MARGIN used during training
SMOOTHING = 0.5         # 0 = no smoothing (raw, jittery), closer to 1 = smoother but laggier
DEFAULT_VALUE = 0.5     # value reported before any hand has ever been seen on that side
MAX_HANDS = 2


# ============================================================
# MODEL DEFINITION (must match training script exactly)
# ============================================================

class ThumbRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.mobilenet_v3_small(weights=None)
        backbone.classifier[-1] = nn.Linear(backbone.classifier[-1].in_features, 1)
        self.backbone = backbone

    def forward(self, x):
        return torch.sigmoid(self.backbone(x)).squeeze(1)


def crop_hand_region(img, landmarks, margin=0.4):
    """Crop tightly around the hand (all 21 landmarks) with padding.
    Matches the cropping logic used in thumb_regressor.py training.
    Returns (crop, box_pixels) where box_pixels = (x1, y1, x2, y2), or
    (None, None) if the crop would be invalid."""
    h, w = img.shape[:2]
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    box_w = x_max - x_min
    box_h = y_max - y_min
    pad_w = max(box_w, 0.05) * margin
    pad_h = max(box_h, 0.05) * margin

    x_min -= pad_w
    x_max += pad_w
    y_min -= pad_h
    y_max += pad_h

    x_min = max(0.0, x_min)
    x_max = min(1.0, x_max)
    y_min = max(0.0, y_min)
    y_max = min(1.0, y_max)

    px_min, px_max = int(x_min * w), int(x_max * w)
    py_min, py_max = int(y_min * h), int(y_max * h)

    if px_max <= px_min or py_max <= py_min:
        return None, None

    crop = img[py_min:py_max, px_min:px_max]
    return crop, (px_min, py_min, px_max, py_max)


def classify_side(landmarks, frame_width):
    """Decide whether a detected hand is on the left or right side of
    the (already mirrored) frame, based on the average x position of
    its landmarks."""
    mean_x = sum(lm.x for lm in landmarks) / len(landmarks)  # normalized 0-1
    return "left" if mean_x < 0.5 else "right"


# ============================================================
# CONTROLLER
# ============================================================

class ThumbController:
    def __init__(self, model_path=MODEL_PATH, camera_index=CAMERA_INDEX,
                 crop_margin=CROP_MARGIN, smoothing=SMOOTHING,
                 default_value=DEFAULT_VALUE):
        self.model_path = model_path
        self.camera_index = camera_index
        self.crop_margin = crop_margin
        self.smoothing = smoothing

        self._lock = threading.Lock()
        self._value = {"left": default_value, "right": default_value}
        self._hand_detected = {"left": False, "right": False}
        self._hand_boxes = {"left": None, "right": None}
        self._running = False
        self._thread = None
        self._latest_frame = None  # last raw (mirrored) webcam frame, for debug display

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.preprocess = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.model = ThumbRegressor()
        state_dict = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        self.mp_hands = mp.solutions.hands

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    # ---- single-hand / legacy-friendly accessors ----

    def get_value(self, side=None):
        """With no argument: returns whichever hand is currently seen
        (right takes priority if both are present) -- convenient for
        existing single-hand games. Pass side="left" or "right" for
        two-hand games."""
        with self._lock:
            if side in ("left", "right"):
                return self._value[side]
            if self._hand_detected["right"]:
                return self._value["right"]
            if self._hand_detected["left"]:
                return self._value["left"]
            return self._value["right"]

    def get_hand_detected(self, side=None):
        with self._lock:
            if side in ("left", "right"):
                return self._hand_detected[side]
            return self._hand_detected["left"] or self._hand_detected["right"]

    # ---- explicit two-hand accessors ----

    def get_left_value(self):
        return self.get_value("left")

    def get_right_value(self):
        return self.get_value("right")

    def get_left_detected(self):
        return self.get_hand_detected("left")

    def get_right_detected(self):
        return self.get_hand_detected("right")

    def get_hand_boxes(self):
        """Returns {"left": (x1,y1,x2,y2) or None, "right": (...) or None}
        in pixel coordinates of the raw camera frame -- useful for
        drawing overlays in a debug/preview window."""
        with self._lock:
            return dict(self._hand_boxes)

    def get_latest_frame(self):
        """Returns the most recent raw (already mirrored) webcam frame
        (BGR numpy array) for debug display, or None if not available yet."""
        with self._lock:
            return self._latest_frame

    # ---- background thread ----

    def _run(self):
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            print(f"[ThumbController] Could not open webcam index {self.camera_index}")
            self._running = False
            return

        hands_detector = self.mp_hands.Hands(
            static_image_mode=False,  # video mode: faster, tracks between frames
            max_num_hands=MAX_HANDS,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        smoothed = {"left": self._value["left"], "right": self._value["right"]}

        while self._running:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            frame = cv2.flip(frame, 1)  # mirror, feels natural
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands_detector.process(frame_rgb)

            found = {"left": False, "right": False}
            boxes = {"left": None, "right": None}

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    lm = hand_landmarks.landmark
                    side = classify_side(lm, frame.shape[1])

                    # if two hands land on the same side (rare/edge case),
                    # keep the first one we already processed for that side
                    if found[side]:
                        continue

                    crop, box_pixels = crop_hand_region(frame, lm, margin=self.crop_margin)
                    if crop is None or crop.size == 0:
                        continue

                    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(crop_rgb)
                    input_tensor = self.preprocess(pil_img).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        raw_value = self.model(input_tensor).item()

                    smoothed[side] = self.smoothing * smoothed[side] + (1 - self.smoothing) * raw_value
                    found[side] = True
                    boxes[side] = box_pixels

            with self._lock:
                self._latest_frame = frame
                for side in ("left", "right"):
                    self._hand_detected[side] = found[side]
                    self._hand_boxes[side] = boxes[side]
                    if found[side]:
                        self._value[side] = smoothed[side]
                    # if not found, keep reporting the last known value

        hands_detector.close()
        cap.release()


# ============================================================
# Standalone test -- run this file directly to sanity check
# ============================================================

if __name__ == "__main__":
    print("[test] Starting ThumbController standalone test (two-hand)...")
    controller = ThumbController()
    controller.start()

    try:
        while True:
            lv, rv = controller.get_left_value(), controller.get_right_value()
            ld, rd = controller.get_left_detected(), controller.get_right_detected()

            def bar(v):
                n = int(v * 20)
                return "#" * n + "-" * (20 - n)

            print(f"\rLEFT [{bar(lv)}] {lv:.2f} {'HAND' if ld else '    '}   "
                  f"RIGHT [{bar(rv)}] {rv:.2f} {'HAND' if rd else '    '}   ",
                  end="", flush=True)
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\n[test] Stopping...")
        controller.stop()