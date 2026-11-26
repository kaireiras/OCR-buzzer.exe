import re
import cv2
import numpy as np

from app.config import (
    COL_X_TOLERANCE,
    CORRECTION_MAP,
    CROP_X_END,
    CROP_X_START,
    CROP_Y_END,
    CROP_Y_START,
    OCR_LANG,
    ROW_Y_TOLERANCE,
    TEXT_DET_LIMIT_SIDE_LEN,
    TEXT_DET_LIMIT_TYPE,
)
from app.device import get_device_info, get_ocr_device
from paddleocr import PaddleOCR

OCR_DEVICE = get_ocr_device()
DEVICE_INFO = get_device_info(OCR_DEVICE)

_ocr_engine: PaddleOCR | None = None


def init_ocr_engine() -> PaddleOCR:
  """Initialize and warm up the singleton PaddleOCR engine."""
  global _ocr_engine
  if _ocr_engine is None:
    # Gunakan text_det_limit_side_len=640 agar cepat tanpa kehilangan ketajaman
    engine = PaddleOCR(
        lang=OCR_LANG,
        device=OCR_DEVICE,
        text_det_limit_side_len=TEXT_DET_LIMIT_SIDE_LEN,
        text_det_limit_type=TEXT_DET_LIMIT_TYPE,
        enable_mkldnn=False,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )
    # Pemanasan model saat server dinyalakan agar request pertama langsung cepat
    engine.predict(np.zeros((200, 200, 3), dtype=np.uint8))
    _ocr_engine = engine
  return _ocr_engine


def get_ocr_engine() -> PaddleOCR:
  """Return the singleton PaddleOCR engine, initializing it on first use."""
  return init_ocr_engine()


def siapkan_kertas_putih(img_bgr: np.ndarray) -> np.ndarray:
  """Kartu Pink -> Red Channel murni (garis vertikal tetap tipis sehingga kolom tidak menyatu).

  Kartu Biru -> Blue Channel + CLAHE ringan (untuk menebalkan tinta pulpen
  pudar).
  """
  rata_biru = np.mean(img_bgr[:, :, 0])
  rata_merah = np.mean(img_bgr[:, :, 2])

  if rata_biru > rata_merah:
    kanal = img_bgr[:, :, 0]  # Kartu Biru
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    kanal = clahe.apply(kanal)
  else:
    kanal = img_bgr[:, :, 2]  # Kartu Pink (murni tanpa CLAHE agar kolom BB & TB terpisah bersih)

  return cv2.cvtColor(kanal, cv2.COLOR_GRAY2BGR)


def bersihkan_angka(teks: str, tipe: str) -> float | int | None:
  """Normalize handwritten OCR numeric strings according to field constraints."""
  if not teks:
    return None

  hasil = "".join(CORRECTION_MAP.get(c, c) for c in teks)
  hasil = re.sub(r"[^0-9.]", "", hasil)
  if not hasil:
    return None

  try:
    nilai = float(hasil)
    if tipe == "umur":
      val = int(round(nilai))
      return val if 0 <= val <= 60 else None
    elif tipe == "bb":
      if 20 <= nilai <= 250 and "." not in hasil:
        nilai = nilai / 10.0
      return nilai if 2.0 <= nilai <= 25.0 else None
    elif tipe == "tb":
      if 450 <= nilai <= 1200 and "." not in hasil:
        nilai = nilai / 10.0
      return nilai if 45.0 <= nilai <= 120.0 else None
  except ValueError:
    return None


def proses_gambar_kia(img_bgr: np.ndarray) -> dict:
  """Extract the latest Posyandu visit record from a KIA/KMS card image."""
  # 1. Crop area tengah tempat tabel berada (25% - 90% tinggi, 12% - 82% lebar)
  h, w = img_bgr.shape[:2]
  y1, y2 = int(h * CROP_Y_START), int(h * CROP_Y_END)
  x1, x2 = int(w * CROP_X_START), int(w * CROP_X_END)
  img_crop = img_bgr[y1:y2, x1:x2]

  # 2. Sesuaikan channel warna kertas (Pink vs Biru)
  img_input = siapkan_kertas_putih(img_crop)
  hasil_ocr = get_ocr_engine().predict(img_input)[0]

  kotak_list = hasil_ocr.get("rec_polys", [])
  teks_list = hasil_ocr.get("rec_texts", [])

  # 3. Cari posisi X & Y header tabel secara dinamis
  kolom_x = {}
  y_header = 0
  for kotak, teks in zip(kotak_list, teks_list):
    t = teks.strip().upper()
    x_tengah = (kotak[0][0] + kotak[1][0]) / 2
    y_tengah = (kotak[0][1] + kotak[2][1]) / 2

    if "TGL" in t or "BULAN" in t:
      kolom_x["tanggal"] = x_tengah
      y_header = max(y_header, y_tengah)
    elif t == "UMUR":
      kolom_x["umur"] = x_tengah
    elif t == "BB":
      kolom_x["bb"] = x_tengah
    elif t == "TB":
      kolom_x["tb"] = x_tengah

  if not kolom_x:
    raise ValueError(
        "Header tabel kartu KIA tidak terdeteksi. Pastikan foto tegak dan"
        " fokus."
    )

  # 4. Kelompokkan semua baris valid dari atas ke bawah
  batas_kanan = kolom_x.get("tb", img_crop.shape[1] * 0.85) + 40
  item_tabel = []
  for kotak, teks in zip(kotak_list, teks_list):
    x_tengah = (kotak[0][0] + kotak[1][0]) / 2
    y_tengah = (kotak[0][1] + kotak[2][1]) / 2
    if y_tengah > y_header + 8 and x_tengah < batas_kanan:
      item_tabel.append({"x": x_tengah, "y": y_tengah, "teks": teks.strip()})

  item_tabel.sort(key=lambda item: item["y"])

  baris_list = []
  for item in item_tabel:
    if not baris_list or abs(item["y"] - baris_list[-1][0]["y"]) > ROW_Y_TOLERANCE:
      baris_list.append([item])
    else:
      baris_list[-1].append(item)

  riwayat_valid = []
  for baris in baris_list:
    data_baris = {"tanggal": None, "umur": None, "bb": None, "tb": None}
    for item in baris:
      kolom_terdekat = min(
          kolom_x.keys(), key=lambda k: abs(item["x"] - kolom_x[k])
      )
      if abs(item["x"] - kolom_x[kolom_terdekat]) < COL_X_TOLERANCE:
        data_baris[kolom_terdekat] = item["teks"]

    punya_tgl = data_baris["tanggal"] and re.search(
        r"\d{1,2}[._\-·]\d{1,2}", data_baris["tanggal"]
    )
    jumlah_isi = sum(
        1 for v in [data_baris["umur"], data_baris["bb"], data_baris["tb"]] if v
    )

    if punya_tgl or jumlah_isi >= 2:
      riwayat_valid.append({
          "tanggal": (
              re.sub(r"[._·]", "-", data_baris["tanggal"])
              if data_baris["tanggal"]
              else None
          ),
          "umur": bersihkan_angka(data_baris["umur"], "umur"),
          "bb": bersihkan_angka(data_baris["bb"], "bb"),
          "tb": bersihkan_angka(data_baris["tb"], "tb"),
          "mentah": data_baris,
      })

  if not riwayat_valid:
    raise ValueError("Baris catatan kunjungan tidak ditemukan.")

  terbaru = riwayat_valid[-1]
  sebelumnya = riwayat_valid[-2] if len(riwayat_valid) >= 2 else None

  # 5. Validasi konteks bulan sebelumnya
  if terbaru["umur"] is None and sebelumnya and sebelumnya["umur"] is not None:
    terbaru["umur"] = sebelumnya["umur"] + 1

  if (
      terbaru["tb"] is None
      and sebelumnya
      and sebelumnya["tb"] is not None
      and terbaru["mentah"]["tb"]
  ):
    angka_mentah = re.sub(r"[^0-9]", "", terbaru["mentah"]["tb"])
    digit_awal_lalu = str(int(sebelumnya["tb"]))[0]
    if len(angka_mentah) == 2 and angka_mentah[0] in ["2", "5", "8"]:
      kandidat_tb = float(digit_awal_lalu + angka_mentah[1])
      if sebelumnya["tb"] <= kandidat_tb <= sebelumnya["tb"] + 4.0:
        terbaru["tb"] = kandidat_tb

  perlu_konfirmasi = any(
      terbaru[k] is None for k in ["tanggal", "umur", "bb", "tb"]
  )

  return {
      "tanggal_kunjungan": terbaru["tanggal"],
      "umur_bulan": terbaru["umur"],
      "berat_badan_kg": terbaru["bb"],
      "tinggi_badan_cm": terbaru["tb"],
      "perlu_konfirmasi": perlu_konfirmasi,
      "mentah_ocr": terbaru["mentah"],
  }
