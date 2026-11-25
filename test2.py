
import re
import cv2
import numpy as np
from rapidocr_onnxruntime import RapidOCR

# =========================================================
# KONFIGURASI
# =========================================================

ocr = RapidOCR(text_score=0.1)

TARGET_TANGGAL = "15-10-26"

# Rentang nilai yang diperbolehkan
RENTANG_BB = (2.0, 25.0)
RENTANG_TB = (45.0, 120.0)


# =========================================================
# NORMALISASI ANGKA
# =========================================================

def normalisasi_angka(teks, min_val, max_val):
    cleaned = (
        teks.replace(",", ".")
        .replace("O", "0")
        .replace("o", "0")
        .replace("l", "1")
        .replace("I", "1")
        .replace("S", "5")
        .replace("s", "5")
        .replace("B", "8")
    )

    match = re.search(r"\d+(\.\d+)?", cleaned)

    if not match:
        return None

    nilai = float(match.group(0))

    if min_val <= nilai <= max_val:
        return nilai

    return None


# =========================================================
# PREPROCESSING
# =========================================================

def buat_variasi(gambar):
    """
    Menghasilkan beberapa variasi preprocessing
    untuk dibandingkan oleh RapidOCR.
    """

    variasi = {}

    # 1. Original grayscale yang diperbesar
    original = cv2.resize(
        gambar,
        None,
        fx=4,
        fy=4,
        interpolation=cv2.INTER_CUBIC
    )

    variasi["01_original"] = original

    # 2. CLAHE
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    kontras = clahe.apply(original)

    variasi["02_clahe"] = kontras

    # 3. Denoising
    denoise = cv2.fastNlMeansDenoising(
        kontras,
        None,
        h=10,
        templateWindowSize=7,
        searchWindowSize=21
    )

    variasi["03_denoise"] = denoise

    # 4. Adaptive Threshold
    adaptive = cv2.adaptiveThreshold(
        denoise,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        21,
        10
    )

    variasi["04_adaptive"] = adaptive

    # 5. Otsu Threshold
    _, otsu = cv2.threshold(
        denoise,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    variasi["05_otsu"] = otsu

    # 6. Sharpening
    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])

    sharpen = cv2.filter2D(original, -1, kernel)

    variasi["06_sharpen"] = sharpen

    # Tambahkan padding untuk setiap variasi
    hasil = {}

    for nama, img in variasi.items():
        padded = cv2.copyMakeBorder(
            img,
            30, 30, 40, 40,
            cv2.BORDER_CONSTANT,
            value=255
        )

        hasil[nama] = padded

    return hasil


# =========================================================
# OCR MULTI VARIASI
# =========================================================

def jalankan_ocr(gambar, nama_kolom):
    variasi = buat_variasi(gambar)

    kandidat = []

    print(f"\n========== OCR {nama_kolom} ==========")

    for nama, img in variasi.items():

        # Simpan gambar preprocessing
        cv2.imwrite(
            f"debug_{nama_kolom}_{nama}.jpg",
            img
        )

        # RapidOCR recognition only
        # Convert grayscale ke BGR agar input konsisten
        img_bgr = cv2.cvtColor(
            img,
            cv2.COLOR_GRAY2BGR
        )

        hasil, _ = ocr(
            img_bgr,
            use_det=False,
            use_cls=False,
            use_rec=True
        )

        teks = ""
        skor = 0.0

        if hasil:
            teks = " ".join(
                item[0] for item in hasil
            )

            skor = max(
                float(item[1]) for item in hasil
            )

        print(
            f"{nama:20s} | "
            f"Teks: '{teks}' | "
            f"Skor: {skor:.4f}"
        )

        kandidat.append({
            "metode": nama,
            "teks": teks,
            "skor": skor
        })

    # Pilih hasil dengan skor tertinggi yang tidak kosong
    valid = [
        item for item in kandidat
        if item["teks"].strip()
    ]

    if not valid:
        return {
            "teks": "",
            "skor": 0.0,
            "metode": None
        }

    terbaik = max(
        valid,
        key=lambda item: item["skor"]
    )

    print(
        f"\nHasil terbaik {nama_kolom}: "
        f"{terbaik['teks']} "
        f"(metode: {terbaik['metode']})"
    )

    return terbaik


# =========================================================
# LOAD IMAGE
# =========================================================

img = cv2.imread("test.jpeg")

if img is None:
    raise FileNotFoundError("File test.jpeg tidak ditemukan.")

red_channel = img[:, :, 2]

h_img, w_img = red_channel.shape


# =========================================================
# DETEKSI TANGGAL
# =========================================================

hasil_penuh, _ = ocr(red_channel)

kotak_target = None

if hasil_penuh:
    for item in hasil_penuh:
        teks = item[1].replace(" ", "")

        if TARGET_TANGGAL in teks:
            kotak_target = np.array(
                item[0],
                dtype=np.int32
            )
            break

if kotak_target is None:
    raise ValueError(
        f"Tanggal {TARGET_TANGGAL} tidak ditemukan."
    )


# =========================================================
# CROPPING KOLOM
# =========================================================

x_min, y_min = kotak_target.min(axis=0)
x_max, y_max = kotak_target.max(axis=0)

lebar_tgl = x_max - x_min
tinggi_tgl = y_max - y_min

# Area vertikal baris yang sama
y1 = max(
    0,
    int(y_min - tinggi_tgl * 0.05)
)

y2 = min(
    h_img,
    int(y_max + tinggi_tgl * 0.05)
)

# Kolom berat badan
bb_x1 = max(
    0,
    int(x_max + lebar_tgl * 0.80)
)

bb_x2 = min(
    w_img,
    int(x_max + lebar_tgl * 1.32)
)

# Kolom tinggi badan
tb_x1 = max(
    0,
    int(x_max + lebar_tgl * 1.48)
)

tb_x2 = min(
    w_img,
    int(x_max + lebar_tgl * 1.98)
)

# Validasi koordinat
if bb_x2 <= bb_x1 or tb_x2 <= tb_x1 or y2 <= y1:
    raise ValueError("Area cropping tidak valid.")

img_bb = red_channel[y1:y2, bb_x1:bb_x2]
img_tb = red_channel[y1:y2, tb_x1:tb_x2]

# Simpan hasil crop awal
cv2.imwrite("debug_crop_bb.jpg", img_bb)
cv2.imwrite("debug_crop_tb.jpg", img_tb)

print("Ukuran crop BB:", img_bb.shape)
print("Ukuran crop TB:", img_tb.shape)


# =========================================================
# EKSTRAKSI OCR
# =========================================================

hasil_bb = jalankan_ocr(img_bb, "bb")
hasil_tb = jalankan_ocr(img_tb, "tb")


# =========================================================
# NORMALISASI HASIL
# =========================================================

berat = normalisasi_angka(
    hasil_bb["teks"],
    *RENTANG_BB
)

tinggi = normalisasi_angka(
    hasil_tb["teks"],
    *RENTANG_TB
)


# =========================================================
# OUTPUT
# =========================================================

print("\n========================================")
print("HASIL EKSTRAKSI BARIS", TARGET_TANGGAL)
print("========================================")

print({
    "tanggal": TARGET_TANGGAL,
    "berat_badan_kg": berat,
    "tinggi_badan_cm": tinggi
})

print("\nFile debug berhasil disimpan.")

