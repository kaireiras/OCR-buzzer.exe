import os
import site
import sysconfig

# Set safe environment defaults before importing Paddle/PaddleOCR
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
os.environ.setdefault("FLAGS_allocator_strategy", "auto_growth")
os.environ.setdefault("FLAGS_use_mkldnn", "0")

_DLL_HANDLES = []
_DLL_DIRS_ADDED = set()


def configure_gpu_dll_paths() -> list[str]:
  """Add NVIDIA CUDA/cuDNN wheel bin directories on Windows if present."""
  if os.name != "nt":
    return []

  candidate_roots = []
  try:
    candidate_roots.extend(site.getsitepackages())
  except Exception:
    pass
  purelib = sysconfig.get_paths().get("purelib")
  if purelib:
    candidate_roots.append(purelib)

  added = []
  for root in dict.fromkeys(candidate_roots):
    nvidia_dir = os.path.join(root, "nvidia")
    if not os.path.isdir(nvidia_dir):
      continue
    for entry in os.listdir(nvidia_dir):
      bin_dir = os.path.join(nvidia_dir, entry, "bin")
      norm_dir = os.path.normcase(os.path.abspath(bin_dir))
      if os.path.isdir(bin_dir) and norm_dir not in _DLL_DIRS_ADDED:
        if hasattr(os, "add_dll_directory"):
          _DLL_HANDLES.append(os.add_dll_directory(bin_dir))
        current_path = os.environ.get("PATH", "")
        if bin_dir not in current_path:
          os.environ["PATH"] = bin_dir + os.pathsep + current_path
        _DLL_DIRS_ADDED.add(norm_dir)
        added.append(bin_dir)
  return added


def get_ocr_device() -> str:
  """Return 'gpu:0' when Paddle CUDA is available (unless overridden by OCR_DEVICE), else 'cpu'."""
  override = os.getenv("OCR_DEVICE", "").strip().lower()
  if override:
    if override.startswith("gpu"):
      configure_gpu_dll_paths()
    return override

  configure_gpu_dll_paths()
  try:
    import paddle

    if (
        paddle.device.is_compiled_with_cuda()
        and paddle.device.cuda.device_count() > 0
    ):
      return "gpu:0"
  except Exception:
    pass
  return "cpu"


def get_device_info(device: str | None = None) -> dict:
  """Return active OCR device metadata and CUDA availability."""
  dev = device or get_ocr_device()
  cuda_available = False
  gpu_name = None

  try:
    import paddle

    cuda_available = bool(
        paddle.device.is_compiled_with_cuda()
        and paddle.device.cuda.device_count() > 0
    )
    if cuda_available:
      gpu_idx = 0
      if dev.startswith("gpu:"):
        try:
          gpu_idx = int(dev.split(":", 1)[1].split(",")[0])
        except ValueError:
          gpu_idx = 0
      gpu_name = paddle.device.cuda.get_device_name(gpu_idx)
  except Exception:
    pass

  return {
      "device": dev,
      "cuda_available": cuda_available,
      "gpu_name": gpu_name,
  }
