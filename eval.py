import os
import time
import cv2

# Impor fungsi ekstraksi dan info perangkat langsung dari main.py Anda
from main import DEVICE_INFO, OCR_DEVICE, proses_gambar_kia


def hitung_levenshtein(ref: str, hyp: str) -> int:
  """Menghitung jarak edit (Substitusi + Delesi + Insersi) antara ground truth (ref)

  dan hasil prediksi (hyp) menggunakan Dynamic Programming.
  """
  m, n = len(ref), len(hyp)
  dp = [[0] * (n + 1) for _ in range(m + 1)]

  for i in range(m + 1):
    dp[i][0] = i
  for j in range(n + 1):
    dp[0][j] = j

  for i in range(1, m + 1):
    for j in range(1, n + 1):
      biaya = 0 if ref[i - 1] == hyp[j - 1] else 1
      dp[i][j] = min(
          dp[i - 1][j] + 1,  # Deletion (Penghapusan)
          dp[i][j - 1] + 1,  # Insertion (Penyisipan)
          dp[i - 1][j - 1] + biaya,  # Substitution (Substitusi)
      )
  return dp[m][n]


def hitung_cer(ground_truth: str, prediksi: str) -> tuple[float, int, int]:
  """Mengembalikan (nilai_CER_persen, jumlah_edit_error, panjang_ground_truth)."""
  ref = str(ground_truth).strip()
  hyp = "" if prediksi is None else str(prediksi).strip()

  if len(ref) == 0:
    return (0.0 if len(hyp) == 0 else 100.0), len(hyp), 0

  errors = hitung_levenshtein(ref, hyp)
  cer = (errors / len(ref)) * 100.0
  return cer, errors, len(ref)


# Sesuaikan daftar gambar uji dan ground truth tulisan tangan yang sebenarnya
DATASET_UJI = [
    {
        "file": "WhatsApp Image 2026-11-10 at 08.27.51.jpeg",
        "kategori": "Kartu Pink (Baris ke-11)",
        "ground_truth": {
            "tanggal": "15-10-26",
            "umur": "11",
            "bb": "7.4",
            "tb": "68.0",
        },
    },
    {
        "file": "testing2.jpeg",
        "kategori": "Kartu Pink (Baris ke-22)",
        "ground_truth": {
            "tanggal": "12-11-26",
            "umur": "27",
            "bb": "10.3",
            "tb": "83.9",
        },
    },
    {
        "file": "testing.jpeg",
        "kategori": "Kartu Biru (Baris ke-20, tinta pudar)",
        "ground_truth": {
            "tanggal": "12-11-26",
            "umur": "20",
            "bb": "11.6",
            "tb": "80.0",
        },
    },
]


def jalankan_evaluasi_cer():
  total_sampel = 0
  total_karakter_gt = 0
  total_error_raw = 0
  total_error_final = 0

  error_per_field = {"tanggal": 0, "umur": 0, "bb": 0, "tb": 0}
  karakter_per_field = {"tanggal": 0, "umur": 0, "bb": 0, "tb": 0}
  daftar_latensi = []

  print("=" * 65)
  print("EVALUASI CHARACTER ERROR RATE (CER) - PADDLEOCR KIA PIPELINE")
  gpu_label = f" ({DEVICE_INFO['gpu_name']})" if DEVICE_INFO.get("gpu_name") else ""
  print(f"Perangkat Inferensi: {OCR_DEVICE}{gpu_label}")
  print("=" * 65)

  for item in DATASET_UJI:
    path = item["file"]
    if not os.path.exists(path):
      print(f"[SKIP] File '{path}' tidak ditemukan di direktori ini.\n")
      continue

    img = cv2.imread(path)
    mulai = time.time()
    try:
      hasil = proses_gambar_kia(img)
    except Exception as e:
      print(f"[ERROR] Gagal memproses '{path}': {e}\n")
      continue
    latensi = round(time.time() - mulai, 2)

    total_sampel += 1
    daftar_latensi.append(latensi)

    gt = item["ground_truth"]
    mentah = hasil["mentah_ocr"]
    final_map = {
        "tanggal": hasil["tanggal_kunjungan"],
        "umur": hasil["umur_bulan"],
        "bb": hasil["berat_badan_kg"],
        "tb": hasil["tinggi_badan_cm"],
    }

    err_sampel_raw = 0
    err_sampel_final = 0
    len_sampel_gt = 0

    print(f"Sampel #{total_sampel}: {path} [{item['kategori']}]")
    print(f"Waktu Proses: {latensi} detik | Perlu Konfirmasi Kader: {hasil['perlu_konfirmasi']}")
    print("-" * 65)
    print(f"{'Atribut':<10} | {'Ground Truth':<12} | {'Raw OCR':<12} | {'Final':<10} | {'CER Final'}")
    print("-" * 65)

    for key in ["tanggal", "umur", "bb", "tb"]:
      gt_val = gt[key]
      raw_val = mentah.get(key)
      fin_val = final_map.get(key)

      _, err_r, panjang_gt = hitung_cer(gt_val, raw_val)
      cer_f, err_f, _ = hitung_cer(gt_val, fin_val)

      err_sampel_raw += err_r
      err_sampel_final += err_f
      len_sampel_gt += panjang_gt

      error_per_field[key] += err_f
      karakter_per_field[key] += panjang_gt

      print(
          f"{key:<10} | {str(gt_val):<12} | {str(raw_val):<12} |"
          f" {str(fin_val):<10} | {cer_f:6.2f}%"
      )

    total_karakter_gt += len_sampel_gt
    total_error_raw += err_sampel_raw
    total_error_final += err_sampel_final

    cer_dok_raw = (err_sampel_raw / len_sampel_gt) * 100
    cer_dok_final = (err_sampel_final / len_sampel_gt) * 100
    print("-" * 65)
    print(
        f"CER Sampel Ini -> Raw OCR: {cer_dok_raw:.2f}% | Setelah"
        f" Post-Processing: {cer_dok_final:.2f}%\n"
    )

  if total_sampel == 0:
    print("Tidak ada gambar yang berhasil dievaluasi.")
    return

  overall_cer_raw = (total_error_raw / total_karakter_gt) * 100
  overall_cer_final = (total_error_final / total_karakter_gt) * 100
  avg_latensi = sum(daftar_latensi) / len(daftar_latensi)

  print("=" * 65)
  print("RINGKASAN HASIL EVALUASI KESELURUHAN (MICRO-AVERAGE CER)")
  print("=" * 65)
  print(f"Total Sampel Kartu Diuji          : {total_sampel} kartu")
  print(f"Total Karakter Ground Truth (N)   : {total_karakter_gt} karakter")
  print(f"CER Mentah (PaddleOCR Murni)      : {overall_cer_raw:.2f}% ({total_error_raw}/{total_karakter_gt} error)")
  print(f"CER Akhir (+ Post-Processing)     : {overall_cer_final:.2f}% ({total_error_final}/{total_karakter_gt} error)")
  print(f"Character Accuracy Akhir (100-CER): {100 - overall_cer_final:.2f}%")
  print(f"Rata-rata Waktu Inferensi ({OCR_DEVICE.upper()}) : {avg_latensi:.2f} detik/gambar")
  print("-" * 65)
  print("Rincian CER Akhir per Kolom:")
  for key in ["tanggal", "umur", "bb", "tb"]:
    cer_col = (error_per_field[key] / karakter_per_field[key]) * 100
    print(
        f"  - {key:<10}: {cer_col:6.2f}%"
        f" ({error_per_field[key]}/{karakter_per_field[key]} karakter salah)"
    )


if __name__ == "__main__":
  jalankan_evaluasi_cer()