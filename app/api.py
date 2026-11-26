from contextlib import asynccontextmanager
import time

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.ocr_pipeline import (
    DEVICE_INFO,
    OCR_DEVICE,
    init_ocr_engine,
    proses_gambar_kia,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
  """Warm up the OCR engine on FastAPI startup."""
  init_ocr_engine()
  yield


app = FastAPI(title="Posyandu Universal Offline OCR API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def health_check():
  return {
      "status": "ready",
      "engine": f"PaddleOCR Fast Universal ({OCR_DEVICE.upper()})",
      "device": DEVICE_INFO["device"],
      "cuda_available": DEVICE_INFO["cuda_available"],
      "gpu_name": DEVICE_INFO["gpu_name"],
  }


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
  except HTTPException:
    raise
  except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))
