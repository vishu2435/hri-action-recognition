# Author: Himashi
from __future__ import annotations

CLASS_NAMES = {
    1: "boxing",
    2: "handclapping",
    3: "handwaving",
    4: "jogging",
    5: "running",
    6: "walking",
}

NAME_TO_ID = {name: class_id for class_id, name in CLASS_NAMES.items()}
FRAME_COUNT = 40
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".pgm"}

