"""Eksperimen ekstraksi angka menggunakan Google Gemini API (memerlukan GEMINI_API_KEY di .env)."""

import json
from pathlib import Path
import time

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types
from PIL import Image
from pydantic import BaseModel, Field

load_dotenv()
client = genai.Client()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_IMAGE = PROJECT_ROOT / "samples" / "potongan_angka.jpeg"


class DataPengukuranBayi(BaseModel):
  berat_badan_kg: float | None = Field(
      description="Berat badan bayi dalam kg, rentang wajar 2.0 sampai 25.0"
  )
  tinggi_badan_cm: float | None = Field(
      description="Tinggi badan bayi dalam cm, rentang wajar 45.0 sampai 120.0"
  )
  tingkat_keyakinan: str = Field(
      description="Pilih: 'tinggi', 'sedang', atau 'rendah' berdasarkan kejelasan tulisan"
  )


gambar = Image.open(SAMPLE_IMAGE)

prompt = """
Ini adalah potongan gambar tulisan tangan catatan pengukuran bayi:
- Berat badan dalam satuan kg (rentang 2.0 - 25.0)
- Tinggi badan dalam satuan cm (rentang 45.0 - 120.0)

Ekstrak kedua angka tersebut ke dalam JSON. 
Catatan penting:
- Tulisan tangan di Indonesia sering memakai koma (misal '7,4' atau '68,5'), ubah menjadi titik desimal (7.4 dan 68.5).
- Jika coretan tidak terbaca atau kosong, isi null.
"""

# Coba maksimal 3 kali jika server mengalami 503 (high demand)
for percobaan in range(1, 4):
  try:
    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents=[gambar, prompt],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=DataPengukuranBayi,
            temperature=0.0,
        ),
    )
    hasil = json.loads(response.text)
    print("Berhasil:", hasil)
    break  # Keluar dari loop jika sudah berhasil

  except errors.ServerError as e:
    if e.code == 503 and percobaan < 3:
      jeda = percobaan * 3  # Tunggu 3 detik, lalu 6 detik
      print(
          f"Server sedang sibuk (503). Mencoba ulang dalam {jeda} detik..."
          f" ({percobaan}/3)"
      )
      time.sleep(jeda)
    else:
      print("Gagal setelah 3 kali percobaan:", e)
