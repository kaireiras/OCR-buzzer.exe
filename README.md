# Posyandu Universal Offline OCR API (`OCR-buzzer.exe`)

API dan pipeline OCR offline berbasis **PaddleOCR** dan **FastAPI** untuk mengekstrak data kunjungan terbaru tulisan tangan dari foto **Kartu KIA / KMS** (Tanggal Kunjungan, Umur dalam bulan, Berat Badan dalam kg, dan Tinggi Badan dalam cm).

## Fitur Utama

- **Deteksi Warna Kartu Otomatis (Pink & Biru):** Memilih kanal warna optimal secara otomatis (Red channel murni untuk kartu Pink; Blue channel + CLAHE ringan untuk kartu Biru dengan tinta pudar).
- **Parsing Tabel Dinamis:** Mendeteksi posisi kolom `TGL/BULAN`, `UMUR`, `BB`, dan `TB`, mengelompokkan item per baris, lalu mengambil baris kunjungan terbaru.
- **Post-Processing Tulisan Tangan & Validasi Konteks:**
  - Mengoreksi huruf yang sering tertukar dengan angka (misalnya `Z`/`T` → `7`, `O`/`D` → `0`, `S` → `5`, `B` → `8`, `,`/`·` → `.`).
  - Menormalisasi angka tanpa koma (misalnya `74` → `7.4` kg atau `680` → `68.0` cm) sesuai rentang fisiologis anak (`umur`: 0–60 bulan, `bb`: 2.0–25.0 kg, `tb`: 45.0–120.0 cm).
  - Melakukan fallback berbasis riwayat baris sebelumnya apabila umur atau digit awal tinggi badan pada baris terakhir terpotong/kabur, serta menandai `perlu_konfirmasi: true` jika masih ada kolom yang kosong.
- **Akselerasi GPU Otomatis + Fallback CPU:** Mendeteksi CUDA/GPU secara otomatis (termasuk registrasi direktori DLL CUDA/cuDNN di Windows) dengan fallback aman ke CPU, atau dapat dipaksa lewat environment variable `OCR_DEVICE`.
- **Evaluasi CER Bawaan:** Tersedia skrip evaluasi Character Error Rate (CER) menggunakan jarak Levenshtein untuk mengukur akurasi raw OCR vs hasil post-processing.

---

## Struktur Direktori

```text
OCR-buzzer.exe/
├── README.md
├── requirements.txt
├── .gitignore
├── main.py                      # Entrypoint root (re-export FastAPI app & helper utama)
├── app/
│   ├── __init__.py
│   ├── config.py                # Env defaults Paddle/OMP/MKL + parameter crop, toleransi, & peta koreksi
│   ├── device.py                # Deteksi GPU CUDA / fallback CPU & konfigurasi DLL Windows
│   ├── ocr_pipeline.py          # Singleton PaddleOCR + warmup, preprocessing, & parsing tabel KIA
│   └── api.py                   # Aplikasi FastAPI (lifespan warmup, CORS, GET /, POST /scan-kia)
├── scripts/
│   ├── eval_cer.py              # Skrip evaluasi Character Error Rate (CER)
│   └── parser_kia.py            # CLI untuk mengekstrak 1 foto kartu KIA menjadi JSON
├── experiments/
│   ├── gemini_ocr.py            # Eksperimen OCR dengan Google Gemini API
│   ├── rapidocr_exp.py          # Eksperimen variasi preprocessing dengan RapidOCR
│   ├── easyocr_exp.py           # Eksperimen recognition angka kotak dengan EasyOCR
│   └── paddle_raw_debug.py      # Skrip debug bounding box & skor mentah PaddleOCR
└── samples/
    ├── kartu_pink_baris11.jpeg  # Sampel kartu KIA Pink (baris ke-11)
    ├── kartu_pink_baris22.jpeg  # Sampel kartu KIA Pink (baris ke-22)
    ├── kartu_biru_baris20.jpeg  # Sampel kartu Biru (baris ke-20, tinta pudar)
    ├── kartu_sample_lain.jpeg   # Sampel kartu tambahan
    └── potongan_angka.jpeg      # Potongan gambar angka tulisan tangan
```

---

## Instalasi & Konfigurasi

Prasyarat: **Python 3.10+** (diuji pada Python 3.13 dengan `paddleocr 3.7.0` dan `paddlepaddle-gpu 3.3.1`).

1. **Buat dan aktifkan virtual environment (disarankan):**

   ```bash
   python -m venv .venv
   # Windows PowerShell:
   .venv\Scripts\Activate.ps1
   # Linux / macOS:
   source .venv/bin/activate
   ```

2. **Install dependensi utama:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Install backend PaddlePaddle (pilih salah satu sesuai perangkat):**

   - **Mode GPU (NVIDIA CUDA):**
     ```bash
     pip install paddlepaddle-gpu
     ```
   - **Mode CPU only:**
     ```bash
     pip install paddlepaddle
     ```

4. **Konfigurasi perangkat inferensi (opsional):**
   Secara default `app/device.py` akan memilih `gpu:0` jika Paddle CUDA tersedia, atau `cpu` jika tidak. Anda bisa menimpa pilihan tersebut melalui environment variable `OCR_DEVICE`:

   ```powershell
   # Paksa berjalan di CPU
   $env:OCR_DEVICE="cpu"

   # Paksa GPU tertentu
   $env:OCR_DEVICE="gpu:0"
   ```

---

## Menjalankan Server API

Jalankan server dari direktori root project:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Atau jalankan langsung via Python:

```bash
python main.py
```

Setelah server aktif, dokumentasi interaktif Swagger UI tersedia di:
- `http://localhost:8000/docs`

---

## Dokumentasi Endpoint

### `GET /`

Memeriksa kesiapan server serta perangkat inferensi (CPU/GPU) yang sedang aktif.

**Contoh Response (`200 OK`):**

```json
{
  "status": "ready",
  "engine": "PaddleOCR Fast Universal (GPU:0)",
  "device": "gpu:0",
  "cuda_available": true,
  "gpu_name": "NVIDIA GeForce RTX 3050 6GB Laptop GPU"
}
```

### `POST /scan-kia`

Menerima unggahan gambar kartu KIA/KMS (`multipart/form-data` dengan field `file`) dan mengembalikan hasil ekstraksi baris kunjungan terbaru.

**Contoh Permintaan (`curl`):**

```bash
curl -X POST "http://localhost:8000/scan-kia" \
  -F "file=@samples/kartu_pink_baris11.jpeg"
```

**Contoh Response Sukses (`200 OK`):**

```json
{
  "status": "success",
  "waktu_proses_detik": 1.53,
  "data": {
    "tanggal_kunjungan": "15-10-26",
    "umur_bulan": 11,
    "berat_badan_kg": 7.4,
    "tinggi_badan_cm": 68.0,
    "perlu_konfirmasi": false,
    "mentah_ocr": {
      "tanggal": "15-10-26",
      "umur": "11",
      "bb": "Z.4",
      "tb": "68"
    }
  }
}
```

**Kode Error:**
- `400 Bad Request` — File gambar tidak valid atau gagal di-decode.
- `422 Unprocessable Entity` — Header tabel atau baris catatan kunjungan pada kartu KIA tidak terdeteksi.
- `500 Internal Server Error` — Kesalahan tak terduga saat pemrosesan.

---

## CLI Parser & Evaluasi CER

### 1. Ekstraksi 1 Gambar via CLI

```bash
# Menggunakan gambar sampel default (samples/kartu_pink_baris11.jpeg)
python scripts/parser_kia.py

# Atau menentukan path gambar lain
python scripts/parser_kia.py samples/kartu_pink_baris22.jpeg
```

### 2. Evaluasi Akurasi (Character Error Rate)

```bash
python scripts/eval_cer.py
```

Ringkasan hasil evaluasi pada `gpu:0` (NVIDIA GeForce RTX 3050 6GB Laptop GPU):

| Sampel | Kategori | CER Raw OCR | CER Setelah Post-Processing | Status |
| :--- | :--- | :---: | :---: | :--- |
| `kartu_pink_baris11.jpeg` | Kartu Pink (Baris ke-11) | 17.65% | **0.00%** | Berhasil |
| `kartu_pink_baris22.jpeg` | Kartu Pink (Baris ke-22) | 16.67% | **5.56%** | Berhasil |
| `kartu_biru_baris20.jpeg` | Kartu Biru (Baris ke-20, tinta pudar) | — | — | Header tabel tidak terdeteksi |

**Ringkasan Hasil Evaluasi Keseluruhan (Micro-Average CER):**

- **Total Sampel Kartu Diuji:** `2 kartu`
- **Total Karakter Ground Truth (N):** `35 karakter`
- **CER Mentah (PaddleOCR Murni):** `17.14%` (`6/35` error)
- **CER Akhir (+ Post-Processing):** `2.86%` (`1/35` error)
- **Character Accuracy Akhir (`100 - CER`):** `97.14%`
- **Rata-rata Waktu Inferensi (`GPU:0`):** `1.16 detik/gambar`

**Rincian CER Akhir per Kolom:**

| Kolom | CER Akhir | Jumlah Karakter Salah |
| :--- | :---: | :---: |
| `tanggal` | `0.00%` | 0/16 karakter |
| `umur` | `0.00%` | 0/4 karakter |
| `bb` | `0.00%` | 0/7 karakter |
| `tb` | `12.50%` | 1/8 karakter |

---

## Folder `experiments/`

Berisi skrip-skrip eksplorasi metode OCR alternatif dan debugging selama pengembangan:

- `experiments/gemini_ocr.py` — Uji ekstraksi terstruktur memakai Google Gemini API (`GEMINI_API_KEY` di `.env`).
- `experiments/rapidocr_exp.py` — Eksperimen multi-variasi preprocessing (CLAHE, denoise, adaptive/Otsu threshold, sharpen) dengan RapidOCR.
- `experiments/easyocr_exp.py` — Uji pengenalan angka terisolasi dengan EasyOCR pada potongan debug kolom BB/TB.
- `experiments/paddle_raw_debug.py` — Menampilkan koordinat bounding box, teks mentah, dan confidence score dari PaddleOCR.
