import os

os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
os.environ["FLAGS_use_mkldnn"] = "0"

import re
import cv2
from ocr_device import get_ocr_device
from paddleocr import PaddleOCR

OCR_DEVICE = get_ocr_device()

ocr = PaddleOCR(
    lang="en",
    device=OCR_DEVICE,
    enable_mkldnn=False,
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
)


def bersihkan_angka_tulisan_tangan(teks: str, tipe: str) -> float | int | None:
  # Koreksi huruf yang sering tertukar dengan angka tulisan tangan
  peta_koreksi = {
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
  hasil = "".join(peta_koreksi.get(c, c) for c in teks)
  hasil = re.sub(r"[^0-9.]", "", hasil)

  if not hasil:
    return None

  try:
    nilai = float(hasil)
    if tipe == "umur":
      return int(round(nilai))
    elif tipe == "bb":
      # Jika terbaca 2 digit tanpa koma (misal '72' padahal 7.2 kg)
      if 20 <= nilai <= 250 and "." not in hasil:
        nilai = nilai / 10.0
      return nilai if 2.0 <= nilai <= 25.0 else None
    elif tipe == "tb":
      # Jika terbaca 3 digit tanpa koma (misal '632' padahal 63.2 cm)
      if 450 <= nilai <= 1200 and "." not in hasil:
        nilai = nilai / 10.0
      return nilai if 45.0 <= nilai <= 120.0 else None
  except ValueError:
    return None


def ekstrak_kartu_kia(img_path: str) -> dict:
  img = cv2.imread(img_path)
  red_channel = img[:, :, 2]
  cv2.imwrite("temp_red.jpg", red_channel)

  hasil_ocr = ocr.predict("temp_red.jpg")[0]
  kotak_list = hasil_ocr.get("rec_polys", [])
  teks_list = hasil_ocr.get("rec_texts", [])

  # 1. Cari posisi X & Y header tabel secara dinamis
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

  # 2. Kumpulkan item di bawah header & kelompokkan per baris (toleransi Y = 14px)
  item_tabel = []
  for kotak, teks in zip(kotak_list, teks_list):
    x_tengah = (kotak[0][0] + kotak[1][0]) / 2
    y_tengah = (kotak[0][1] + kotak[2][1]) / 2
    if y_tengah > y_header + 8 and x_tengah < kolom_x.get("tb", 420) + 40:
      item_tabel.append({"x": x_tengah, "y": y_tengah, "teks": teks.strip()})

  item_tabel.sort(key=lambda item: item["y"])

  baris_list = []
  for item in item_tabel:
    if not baris_list or abs(item["y"] - baris_list[-1][0]["y"]) > 14:
      baris_list.append([item])
    else:
      baris_list[-1].append(item)

  # 3. Filter hanya baris yang punya tanggal kunjungan valid & ambil yang paling bawah
  baris_valid = []
  for baris in baris_list:
    data_baris = {"tanggal": None, "umur": None, "bb": None, "tb": None}
    for item in baris:
      # Cocokkan item ke kolom header terdekat
      kolom_terdekat = min(
          kolom_x.keys(), key=lambda k: abs(item["x"] - kolom_x[k])
      )
      if abs(item["x"] - kolom_x[kolom_terdekat]) < 35:
        data_baris[kolom_terdekat] = item["teks"]

    # Cek apakah kolom tanggal berisi pola angka tanggal (misal 15.10.26 atau 15-10-26)
    if data_baris["tanggal"] and re.search(
        r"\d{1,2}[.\-·]\d{1,2}", data_baris["tanggal"]
    ):
      baris_valid.append(data_baris)

  if not baris_valid:
    return {"error": "Baris kunjungan tidak ditemukan"}

  terbaru = baris_valid[-1]
  tgl_bersih = re.sub(r"[.·]", "-", terbaru["tanggal"])

  return {
      "tanggal_kunjungan": tgl_bersih,
      "umur_bulan": bersihkan_angka_tulisan_tangan(
          terbaru["umur"] or "", "umur"
      ),
      "berat_badan_kg": bersihkan_angka_tulisan_tangan(
          terbaru["bb"] or "", "bb"
      ),
      "tinggi_badan_cm": bersihkan_angka_tulisan_tangan(
          terbaru["tb"] or "", "tb"
      ),
      "mentah_ocr": terbaru,
  }


if __name__ == "__main__":
  hasil = ekstrak_kartu_kia("WhatsApp Image 2026-11-10 at 08.27.51.jpeg")
  print("=== JSON SIAP PAKAI UNTUK APLIKASI HP ===")
  print(hasil)
