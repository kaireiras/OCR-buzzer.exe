import uvicorn

from app.api import app
from app.ocr_pipeline import (
    DEVICE_INFO,
    OCR_DEVICE,
    bersihkan_angka,
    init_ocr_engine,
    proses_gambar_kia,
    siapkan_kertas_putih,
)

__all__ = [
    "app",
    "OCR_DEVICE",
    "DEVICE_INFO",
    "init_ocr_engine",
    "siapkan_kertas_putih",
    "bersihkan_angka",
    "proses_gambar_kia",
]

if __name__ == "__main__":
  uvicorn.run("main:app", host="0.0.0.0", port=8000)
