import os

# Env defaults before importing Paddle/PaddleOCR to speed up startup and avoid oneDNN issues
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
os.environ.setdefault("FLAGS_allocator_strategy", "auto_growth")
os.environ.setdefault("FLAGS_use_mkldnn", "0")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

# PaddleOCR parameters
OCR_LANG = "en"
TEXT_DET_LIMIT_SIDE_LEN = 640
TEXT_DET_LIMIT_TYPE = "max"

# Relative crop bounds for the KIA table region
CROP_Y_START = 0.25
CROP_Y_END = 0.90
CROP_X_START = 0.12
CROP_X_END = 0.82

# Table row & column clustering tolerances (pixels)
ROW_Y_TOLERANCE = 14
COL_X_TOLERANCE = 35

# Handwritten character OCR confusion corrections
CORRECTION_MAP = {
    "Z": "7",
    "z": "7",
    "T": "7",
    "O": "0",
    "o": "0",
    "D": "0",
    "C": "6",
    "c": "6",
    "G": "6",
    "B": "8",
    "S": "5",
    "s": "5",
    "I": "1",
    "l": "1",
    "|": "1",
    ",": ".",
    "·": ".",
}
