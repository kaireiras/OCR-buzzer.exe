# import os

# # Mempercepat startup & menghindari bug PIR oneDNN di CPU
# os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
# os.environ["FLAGS_use_mkldnn"] = "0"

# import re
# import time
# import cv2
# import numpy as np
# from fastapi import FastAPI, File, HTTPException, UploadFile
# from fastapi.middleware.cors import CORSMiddleware
# from paddleocr import PaddleOCR

# app = FastAPI(title="Posyandu Offline PaddleOCR API")

# # Izinkan akses dari aplikasi HP / Web Mobile saat demo
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Muat model standar sekali saat server startup
# ocr = PaddleOCR(
#     lang="en",
#     enable_mkldnn=False,
#     use_doc_orientation_classify=False,
#     use_doc_unwarping=False,
#     use_textline_orientation=False,
# )

# # Pemanasan model saat server dinyalakan agar request pertama dari HP langsung cepat
# ocr.predict(np.zeros((200, 200, 3), dtype=np.uint8))


# def bersihkan_angka_tulisan_tangan(teks: str, tipe: str) -> float | int | None:
#   peta_koreksi = {
#       "Z": "7",
#       "z": "7",
#       "T": "7",
#       "O": "0",
#       "o": "0",
#       "D": "0",
#       "C": "6",
#       "c": "6",
#       "G": "6",
#       "B": "8",
#       "S": "5",
#       "s": "5",
#       "I": "1",
#       "l": "1",
#       "|": "1",
#       ",": ".",
#       "·": ".",
#   }
#   hasil = "".join(peta_koreksi.get(c, c) for c in teks)
#   hasil = re.sub(r"[^0-9.]", "", hasil)
#   if not hasil:
#     return None

#   try:
#     if tipe == "umur":
#       nilai = float(hasil)
#       return int(round(nilai))

#     elif tipe == "bb":
#       nilai = float(hasil)
#       # Jika terbaca 2 digit tanpa koma (misal '72' padahal 7.2 kg)
#       if 20 <= nilai <= 250 and "." not in hasil:
#         nilai = nilai / 10.0
#       return nilai if 2.0 <= nilai <= 25.0 else None

#     elif tipe == "tb":
#       # Angka 6 tulisan tangan sering terbaca 2 saat lengkungan atasnya tipis (misal '28' -> '68')
#       if hasil.startswith("2") and len(hasil.split(".")[0]) == 2:
#         hasil = "6" + hasil[1:]
#       nilai = float(hasil)
#       # Jika terbaca 3 digit tanpa koma (misal '632' padahal 63.2 cm)
#       if 450 <= nilai <= 1200 and "." not in hasil:
#         nilai = nilai / 10.0
#       return nilai if 45.0 <= nilai <= 120.0 else None

#   except ValueError:
#     return None


# def proses_gambar_kia(img_bgr: np.ndarray) -> dict:
#   # 1. Crop area tengah tempat tabel KIA berada (buang pinggiran meja/kertas kosong)
#   # Ketajaman tulisan tetap 100% asli, tapi area scan CPU berkurang >60%
#   h, w = img_bgr.shape[:2]
#   y1, y2 = int(h * 0.25), int(h * 0.80)
#   x1, x2 = int(w * 0.15), int(w * 0.80)
#   img_crop = img_bgr[y1:y2, x1:x2]

#   # 2. Ambil Red Channel (index 2) agar warna kertas pink hilang
#   red_channel = img_crop[:, :, 2]
#   img_input = cv2.cvtColor(red_channel, cv2.COLOR_GRAY2BGR)
#   hasil_ocr = ocr.predict(img_input)[0]

#   kotak_list = hasil_ocr.get("rec_polys", [])
#   teks_list = hasil_ocr.get("rec_texts", [])

#   # 3. Cari posisi X & Y header tabel secara dinamis
#   kolom_x = {}
#   y_header = 0
#   for kotak, teks in zip(kotak_list, teks_list):
#     t = teks.strip().upper()
#     x_tengah = (kotak[0][0] + kotak[1][0]) / 2
#     y_tengah = (kotak[0][1] + kotak[2][1]) / 2

#     if "TGL" in t or "BULAN" in t:
#       kolom_x["tanggal"] = x_tengah
#       y_header = max(y_header, y_tengah)
#     elif t == "UMUR":
#       kolom_x["umur"] = x_tengah
#     elif t == "BB":
#       kolom_x["bb"] = x_tengah
#     elif t == "TB":
#       kolom_x["tb"] = x_tengah

#   if not kolom_x:
#     raise ValueError(
#         "Header tabel kartu KIA tidak terdeteksi. Pastikan kartu berada di"
#         " tengah layar."
#     )

#   # 4. Kumpulkan item di bawah header & kelompokkan per baris (toleransi Y = 14px)
#   batas_kanan = kolom_x.get("tb", img_crop.shape[1] * 0.85) + 40
#   item_tabel = []
#   for kotak, teks in zip(kotak_list, teks_list):
#     x_tengah = (kotak[0][0] + kotak[1][0]) / 2
#     y_tengah = (kotak[0][1] + kotak[2][1]) / 2
#     if y_tengah > y_header + 8 and x_tengah < batas_kanan:
#       item_tabel.append({"x": x_tengah, "y": y_tengah, "teks": teks.strip()})

#   item_tabel.sort(key=lambda item: item["y"])

#   baris_list = []
#   for item in item_tabel:
#     if not baris_list or abs(item["y"] - baris_list[-1][0]["y"]) > 14:
#       baris_list.append([item])
#     else:
#       baris_list[-1].append(item)

#   # 5. Filter hanya baris dengan tanggal kunjungan valid & ambil baris paling bawah
#   baris_valid = []
#   for baris in baris_list:
#     data_baris = {"tanggal": None, "umur": None, "bb": None, "tb": None}
#     for item in baris:
#       kolom_terdekat = min(
#           kolom_x.keys(), key=lambda k: abs(item["x"] - kolom_x[k])
#       )
#       if abs(item["x"] - kolom_x[kolom_terdekat]) < 35:
#         data_baris[kolom_terdekat] = item["teks"]

#     if data_baris["tanggal"] and re.search(
#         r"\d{1,2}[.\-·]\d{1,2}", data_baris["tanggal"]
#     ):
#       baris_valid.append(data_baris)

#   if not baris_valid:
#     raise ValueError("Baris catatan kunjungan tidak ditemukan.")

#   terbaru = baris_valid[-1]
#   return {
#       "tanggal_kunjungan": re.sub(r"[.·]", "-", terbaru["tanggal"]),
#       "umur_bulan": bersihkan_angka_tulisan_tangan(
#           terbaru["umur"] or "", "umur"
#       ),
#       "berat_badan_kg": bersihkan_angka_tulisan_tangan(
#           terbaru["bb"] or "", "bb"
#       ),
#       "tinggi_badan_cm": bersihkan_angka_tulisan_tangan(
#           terbaru["tb"] or "", "tb"
#       ),
#       "mentah_ocr": terbaru,
#   }


# @app.get("/")
# def health_check():
#   return {"status": "ready", "engine": "PaddleOCR Auto-Crop CPU"}


# @app.post("/scan-kia")
# async def scan_kartu_kia(file: UploadFile = File(...)):
#   mulai = time.time()
#   try:
#     isi_file = await file.read()
#     arr = np.frombuffer(isi_file, np.uint8)
#     img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
#     if img_bgr is None:
#       raise HTTPException(status_code=400, detail="File gambar tidak valid.")

#     data = proses_gambar_kia(img_bgr)
#     durasi = round(time.time() - mulai, 2)
#     return {
#         "status": "success",
#         "waktu_proses_detik": durasi,
#         "data": data,
#     }
#   except ValueError as e:
#     raise HTTPException(status_code=422, detail=str(e))
#   except Exception as e:
#     raise HTTPException(status_code=500, detail=str(e))


# import os

# os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
# os.environ["FLAGS_use_mkldnn"] = "0"
# os.environ["OMP_NUM_THREADS"] = "4"

# import re
# import time
# import cv2
# import numpy as np
# from fastapi import FastAPI, File, HTTPException, UploadFile
# from fastapi.middleware.cors import CORSMiddleware
# from paddleocr import PaddleOCR

# app = FastAPI(title="Posyandu Universal Offline OCR API")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# ocr = PaddleOCR(
#     lang="en",
#     enable_mkldnn=False,
#     use_doc_orientation_classify=False,
#     use_doc_unwarping=False,
#     use_textline_orientation=False,
# )

# # Pemanasan model saat server startup
# ocr.predict(np.zeros((200, 200, 3), dtype=np.uint8))


# def siapkan_kertas_putih_kontras(img_bgr: np.ndarray) -> np.ndarray:
#   """
#     Otomatis mendeteksi apakah kartu berwarna Pink (pilih Red Channel)
#     atau Biru (pilih Blue Channel), lalu menajamkan tinta pulpen tipis dengan CLAHE.
#     """
#   rata_biru = np.mean(img_bgr[:, :, 0])
#   rata_merah = np.mean(img_bgr[:, :, 2])

#   # Pilih channel dengan latar paling terang agar warna kertas hilang
#   if rata_biru > rata_merah:
#     kanal_terang = img_bgr[:, :, 0]  # Kartu Biru
#   else:
#     kanal_terang = img_bgr[:, :, 2]  # Kartu Pink

#   # Tingkatkan kontras lokal (CLAHE) agar tulisan pulpen pudar tetap terbaca
#   clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
#   kanal_tajam = clahe.apply(kanal_terang)

#   return cv2.cvtColor(kanal_tajam, cv2.COLOR_GRAY2BGR)


# def bersihkan_angka(teks: str, tipe: str) -> float | int | None:
#   if not teks:
#     return None
#   peta_koreksi = {
#       "Z": "7",
#       "z": "7",
#       "T": "7",
#       "O": "0",
#       "o": "0",
#       "D": "0",
#       "C": "6",
#       "c": "6",
#       "G": "6",
#       "B": "8",
#       "S": "5",
#       "s": "5",
#       "I": "1",
#       "l": "1",
#       "|": "1",
#       ",": ".",
#       "·": ".",
#   }
#   hasil = "".join(peta_koreksi.get(c, c) for c in teks)
#   hasil = re.sub(r"[^0-9.]", "", hasil)
#   if not hasil:
#     return None

#   try:
#     nilai = float(hasil)
#     if tipe == "umur":
#       val = int(round(nilai))
#       return val if 0 <= val <= 60 else None
#     elif tipe == "bb":
#       if 20 <= nilai <= 250 and "." not in hasil:
#         nilai = nilai / 10.0
#       return nilai if 2.0 <= nilai <= 25.0 else None
#     elif tipe == "tb":
#       if 450 <= nilai <= 1200 and "." not in hasil:
#         nilai = nilai / 10.0
#       return nilai if 45.0 <= nilai <= 120.0 else None
#   except ValueError:
#     return None


# def proses_gambar_kia(img_bgr: np.ndarray) -> dict:
#   # 1. Crop area tengah tabel (diperluas ke 90% tinggi agar baris ke-20 tidak terpotong)
#   h, w = img_bgr.shape[:2]
#   y1, y2 = int(h * 0.25), int(h * 0.90)
#   x1, x2 = int(w * 0.12), int(w * 0.82)
#   img_crop = img_bgr[y1:y2, x1:x2]

#   # 2. Deteksi otomatis warna kartu (Pink/Biru) + penajaman kontras CLAHE
#   img_input = siapkan_kertas_putih_kontras(img_crop)
#   hasil_ocr = ocr.predict(img_input)[0]

#   kotak_list = hasil_ocr.get("rec_polys", [])
#   teks_list = hasil_ocr.get("rec_texts", [])

#   # 3. Cari posisi X & Y header tabel secara dinamis
#   kolom_x = {}
#   y_header = 0
#   for kotak, teks in zip(kotak_list, teks_list):
#     t = teks.strip().upper()
#     x_tengah = (kotak[0][0] + kotak[1][0]) / 2
#     y_tengah = (kotak[0][1] + kotak[2][1]) / 2

#     if "TGL" in t or "BULAN" in t:
#       kolom_x["tanggal"] = x_tengah
#       y_header = max(y_header, y_tengah)
#     elif t == "UMUR":
#       kolom_x["umur"] = x_tengah
#     elif t == "BB":
#       kolom_x["bb"] = x_tengah
#     elif t == "TB":
#       kolom_x["tb"] = x_tengah

#   if not kolom_x:
#     raise ValueError(
#         "Header tabel kartu KIA tidak terdeteksi. Pastikan foto tegak dan"
#         " fokus."
#     )

#   # 4. Kelompokkan semua baris valid dari atas ke bawah
#   batas_kanan = kolom_x.get("tb", img_crop.shape[1] * 0.85) + 40
#   item_tabel = []
#   for kotak, teks in zip(kotak_list, teks_list):
#     x_tengah = (kotak[0][0] + kotak[1][0]) / 2
#     y_tengah = (kotak[0][1] + kotak[2][1]) / 2
#     if y_tengah > y_header + 8 and x_tengah < batas_kanan:
#       item_tabel.append({"x": x_tengah, "y": y_tengah, "teks": teks.strip()})

#   item_tabel.sort(key=lambda item: item["y"])

#   baris_list = []
#   for item in item_tabel:
#     if not baris_list or abs(item["y"] - baris_list[-1][0]["y"]) > 14:
#       baris_list.append([item])
#     else:
#       baris_list[-1].append(item)

#   riwayat_valid = []
#   for baris in baris_list:
#     data_baris = {"tanggal": None, "umur": None, "bb": None, "tb": None}
#     for item in baris:
#       kolom_terdekat = min(
#           kolom_x.keys(), key=lambda k: abs(item["x"] - kolom_x[k])
#       )
#       if abs(item["x"] - kolom_x[kolom_terdekat]) < 35:
#         data_baris[kolom_terdekat] = item["teks"]

#     # Baris dianggap terisi jika punya tanggal ATAU minimal dua angka (umur/bb/tb)
#     punya_tgl = data_baris["tanggal"] and re.search(
#         r"\d{1,2}[.\-·]\d{1,2}", data_baris["tanggal"]
#     )
#     jumlah_isi = sum(
#         1 for v in [data_baris["umur"], data_baris["bb"], data_baris["tb"]] if v
#     )

#     if punya_tgl or jumlah_isi >= 2:
#       riwayat_valid.append({
#           "tanggal": (
#               re.sub(r"[.·]", "-", data_baris["tanggal"])
#               if data_baris["tanggal"]
#               else None
#           ),
#           "umur": bersihkan_angka(data_baris["umur"], "umur"),
#           "bb": bersihkan_angka(data_baris["bb"], "bb"),
#           "tb": bersihkan_angka(data_baris["tb"], "tb"),
#           "mentah": data_baris,
#       })

#   if not riwayat_valid:
#     raise ValueError("Baris catatan kunjungan tidak ditemukan.")

#   terbaru = riwayat_valid[-1]
#   sebelumnya = riwayat_valid[-2] if len(riwayat_valid) >= 2 else None

#   # 5. Validasi cerdas berbasis bulan sebelumnya (tanpa hardcode kaku!)
#   # Jika umur bulan terakhir gagal terbaca tapi bulan sebelumnya terbaca, otomatis +1
#   if terbaru["umur"] is None and sebelumnya and sebelumnya["umur"] is not None:
#     terbaru["umur"] = sebelumnya["umur"] + 1

#   # Jika TB terbaca tidak masuk akal (misal 28 padahal bulan lalu 66), cek kemiripan digit dengan bulan lalu
#   if (
#       terbaru["tb"] is None
#       and sebelumnya
#       and sebelumnya["tb"] is not None
#       and terbaru["mentah"]["tb"]
#   ):
#     angka_mentah = re.sub(r"[^0-9]", "", terbaru["mentah"]["tb"])
#     digit_awal_lalu = str(int(sebelumnya["tb"]))[0]
#     if len(angka_mentah) == 2 and angka_mentah[0] in ["2", "5", "8"]:
#       kandidat_tb = float(digit_awal_lalu + angka_mentah[1])
#       if sebelumnya["tb"] <= kandidat_tb <= sebelumnya["tb"] + 4.0:
#         terbaru["tb"] = kandidat_tb

#   # Tandai jika masih ada kolom yang buram agar UI HP menyorot kolom tersebut untuk dikonfirmasi kader
#   perlu_konfirmasi = any(
#       terbaru[k] is None for k in ["tanggal", "umur", "bb", "tb"]
#   )

#   return {
#       "tanggal_kunjungan": terbaru["tanggal"],
#       "umur_bulan": terbaru["umur"],
#       "berat_badan_kg": terbaru["bb"],
#       "tinggi_badan_cm": terbaru["tb"],
#       "perlu_konfirmasi": perlu_konfirmasi,
#       "mentah_ocr": terbaru["mentah"],
#   }


# @app.get("/")
# def health_check():
#   return {"status": "ready", "engine": "PaddleOCR Universal Pink/Blue CPU"}


# @app.post("/scan-kia")
# async def scan_kartu_kia(file: UploadFile = File(...)):
#   mulai = time.time()
#   try:
#     isi_file = await file.read()
#     arr = np.frombuffer(isi_file, np.uint8)
#     img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
#     if img_bgr is None:
#       raise HTTPException(status_code=400, detail="File gambar tidak valid.")

#     data = proses_gambar_kia(img_bgr)
#     durasi = round(time.time() - mulai, 2)
#     return {
#         "status": "success",
#         "waktu_proses_detik": durasi,
#         "data": data,
#     }
#   except ValueError as e:
#     raise HTTPException(status_code=422, detail=str(e))
#   except Exception as e:
#     raise HTTPException(status_code=500, detail=str(e))

import os

os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import re
import time
import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from paddleocr import PaddleOCR

app = FastAPI(title="Posyandu Universal Offline OCR API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Gunakan text_det_limit_side_len=640 agar cepat di CPU (~3-5 detik) tanpa kehilangan ketajaman
ocr = PaddleOCR(
    lang="en",
    text_det_limit_side_len=640,
    text_det_limit_type="max",
    enable_mkldnn=False,
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
)

# Pemanasan model saat server startup
ocr.predict(np.zeros((200, 200, 3), dtype=np.uint8))


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
  if not teks:
    return None
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
  # 1. Crop area tengah tempat tabel berada (25% - 90% tinggi, 12% - 82% lebar)
  h, w = img_bgr.shape[:2]
  y1, y2 = int(h * 0.25), int(h * 0.90)
  x1, x2 = int(w * 0.12), int(w * 0.82)
  img_crop = img_bgr[y1:y2, x1:x2]

  # 2. Sesuaikan channel warna kertas (Pink vs Biru)
  img_input = siapkan_kertas_putih(img_crop)
  hasil_ocr = ocr.predict(img_input)[0]

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
    if not baris_list or abs(item["y"] - baris_list[-1][0]["y"]) > 14:
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
      if abs(item["x"] - kolom_x[kolom_terdekat]) < 35:
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


@app.get("/")
def health_check():
  return {"status": "ready", "engine": "PaddleOCR Fast Universal CPU"}


@app.post("/scan-kia")
async def scan_kartu_kia(file: UploadFile = File(...)):
  mulai = time.time()
  try:
    isi_file = await file.read()
    arr = np.frombuffer(isi_file, np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
      raise HTTPException(status_code=400, detail="File gambar tidak valid.")

    data = proses_gambar_kia(img_bgr)
    durasi = round(time.time() - mulai, 2)
    return {
        "status": "success",
        "waktu_proses_detik": durasi,
        "data": data,
    }
  except ValueError as e:
    raise HTTPException(status_code=422, detail=str(e))
  except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))