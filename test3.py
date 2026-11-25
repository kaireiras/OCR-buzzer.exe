import re
import cv2
import easyocr

# Inisialisasi EasyOCR (gpu=False aman untuk semua laptop)
reader = easyocr.Reader(["en"], gpu=False)


def parse_angka(teks: str, min_val: float, max_val: float) -> float | None:
  cleaned = teks.replace(",", ".").strip()
  match = re.search(r"\d+(\.\d+)?", cleaned)
  if not match:
    return None
  nilai = float(match.group(0))
  return nilai if (min_val <= nilai <= max_val) else None


def baca_kotak_angka(path_gambar: str) -> tuple[str, float]:
  img = cv2.imread(path_gambar)
  # reader.recognize() langsung membaca seluruh isi kotak tanpa deteksi teks,
  # dan allowlist membatasi output HANYA angka, koma, dan titik.
  hasil = reader.recognize(
      img,
      allowlist="0123456789.,",
      detail=1,
  )
  if not hasil:
    return "", 0.0
  teks, skor = hasil[0][1], float(hasil[0][2])
  return teks, skor


# Uji langsung pada potongan kotak BB (7,4) dan TB (68)
teks_bb, skor_bb = baca_kotak_angka("debug_kolom_bb.jpg")
teks_tb, skor_tb = baca_kotak_angka("debug_kolom_tb.jpg")

print(f"Mentah Kolom BB: '{teks_bb}' (keyakinan: {skor_bb:.2f})")
print(f"Mentah Kolom TB: '{teks_tb}' (keyakinan: {skor_tb:.2f})")

print("\nHasil Ekstraksi:")
print({
    "berat_badan_kg": parse_angka(teks_bb, 2.0, 25.0),
    "tinggi_badan_cm": parse_angka(teks_tb, 45.0, 120.0),
})
