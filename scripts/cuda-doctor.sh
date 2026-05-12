#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ -f "$ROOT_DIR/.venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$ROOT_DIR/.venv/bin/activate"
fi

export CUDA_HOME="${CUDA_HOME:-/nix/store/1mps3cdd4jmzsxcy1nnr18riy62wslsr-cuda-merged-12.9}"
export LD_LIBRARY_PATH="/run/opengl-driver/lib:$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"

python - <<'PY'
import ctypes
import os

print("CUDA_HOME =", os.environ.get("CUDA_HOME"))
print("LD_LIBRARY_PATH =", os.environ.get("LD_LIBRARY_PATH"))

for lib in ("libcuda.so.1", "libcudart.so.12"):
    try:
        ctypes.CDLL(lib)
        print(lib, "OK")
    except OSError as exc:
        print(lib, "FAIL", exc)

try:
    cudart = ctypes.CDLL("libcudart.so.12")
    get_count = cudart.cudaGetDeviceCount
    get_count.argtypes = [ctypes.POINTER(ctypes.c_int)]
    get_count.restype = ctypes.c_int
    get_err = cudart.cudaGetErrorString
    get_err.argtypes = [ctypes.c_int]
    get_err.restype = ctypes.c_char_p

    count = ctypes.c_int()
    err = get_count(ctypes.byref(count))
    msg = get_err(err).decode()
    print("cudaGetDeviceCount err =", err)
    print("cudaGetDeviceCount msg =", msg)
    print("cudaGetDeviceCount count =", count.value)

    if err == 999:
        print("Hint: CUDA runtime sees the driver but UVM may be broken; reloading nvidia_uvm or rebooting often fixes this on NixOS.")
except OSError as exc:
    print("cuda runtime load failed:", exc)
PY
