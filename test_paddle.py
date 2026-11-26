import os

os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
os.environ["FLAGS_use_mkldnn"] = "0"

import cv2
from ocr_device import get_ocr_device
from paddleocr import PaddleOCR

OCR_DEVICE = get_ocr_device()
print(f"Menggunakan perangkat OCR: {OCR_DEVICE}")

# Matikan enable_mkldnn untuk melewati bug PIR oneDNN di PaddlePaddle 3.x (saat fallback ke CPU)
ocr = PaddleOCR(
    lang="en",
    device=OCR_DEVICE,
    enable_mkldnn=False,
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
)

img_path = "WhatsApp Image 2026-11-10 at 08.27.51.jpeg"
img = cv2.imread(img_path)

if img is None:
  raise FileNotFoundError(
      f"File '{img_path}' tidak ditemukan di folder ini!"
  )

# Hilangkan warna kertas pink dengan mengambil Red Channel (index 2)
red_channel = img[:, :, 2]
cv2.imwrite("temp_red.jpg", red_channel)

hasil = ocr.predict("temp_red.jpg")

print("=== HASIL DETEKSI PADDLEOCR ===")
for res in hasil:
  daftar_teks = res.get("rec_texts", [])
  daftar_skor = res.get("rec_scores", [])
  daftar_kotak = res.get("rec_polys", [])

  gabungan = sorted(
      zip(daftar_kotak, daftar_teks, daftar_skor),
      key=lambda item: (item[0][0][1], item[0][0][0]),
  )

  for kotak, teks, skor in gabungan:
    x_kiri = kotak[0][0]
    y_atas = kotak[0][1]
    print(f"Y:{y_atas:6.1f} | X:{x_kiri:6.1f} | '{teks}' (skor: {skor:.2f})")
