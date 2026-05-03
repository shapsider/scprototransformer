#!/usr/bin/env bash
set -euo pipefail

CONDA_BIN="${CONDA_BIN:-/data/miniconda3/bin/conda}"
ENV_PREFIX="/data/miniconda3/envs/scPT"

if [[ ! -x "${CONDA_BIN}" ]]; then
  echo "Conda not found or not executable: ${CONDA_BIN}" >&2
  echo "Set CONDA_BIN=/path/to/conda and rerun." >&2
  exit 1
fi

echo "Creating scPT environment at: ${ENV_PREFIX}"

"${CONDA_BIN}" create -y -p "${ENV_PREFIX}" python=3.8
"${CONDA_BIN}" install -y -p "${ENV_PREFIX}" \
  pytorch=1.7.1 torchvision=0.8.2 torchaudio=0.7.2 cudatoolkit=11.0 \
  -c pytorch

"${CONDA_BIN}" run -p "${ENV_PREFIX}" python -m pip install \
  "scanpy==1.9.8" tensorboard einops scikit-learn matplotlib pandas scipy numpy h5py pynndescent

echo "Environment ready."
echo "Activate with: conda activate ${ENV_PREFIX}"
echo "Python: ${ENV_PREFIX}/bin/python"
echo
echo "CUDA check:"
LD_LIBRARY_PATH="${ENV_PREFIX}/lib:${LD_LIBRARY_PATH:-}" "${ENV_PREFIX}/bin/python" - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")
PY
