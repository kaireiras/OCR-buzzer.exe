import argparse
import json
from pathlib import Path
import sys

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
  sys.path.insert(0, str(PROJECT_ROOT))

from app.ocr_pipeline import proses_gambar_kia

DEFAULT_SAMPLE = PROJECT_ROOT / "samples" / "kartu_pink_baris11.jpeg"


def main() -> None:
  parser = argparse.ArgumentParser(
      description="Ekstrak baris kunjungan terbaru dari foto kartu KIA/KMS."
  )
  parser.add_argument(
      "image_path",
      nargs="?",
      default=str(DEFAULT_SAMPLE),
      help=f"Path ke file gambar kartu KIA (default: {DEFAULT_SAMPLE})",
  )
  args = parser.parse_args()

  img_path = Path(args.image_path)
  if not img_path.exists():
    raise FileNotFoundError(f"File gambar tidak ditemukan: {img_path}")

  img_bgr = cv2.imread(str(img_path))
  if img_bgr is None:
    raise ValueError(f"File gambar tidak valid atau gagal dibaca: {img_path}")

  hasil = proses_gambar_kia(img_bgr)
  print(json.dumps(hasil, indent=2, ensure_ascii=False))


if __name__ == "__main__":
  main()
