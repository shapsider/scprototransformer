#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/aaa/gelseywang/buddy1/lukatang/scDeepSeek/scProtoTransformer/model/FoudationModel"
ENV_PREFIX="/data/miniconda3/envs/scPT"
PYTHON_BIN="${PYTHON_BIN:-${ENV_PREFIX}/bin/python}"
GPU_ID="${GPU_ID:-7}"
TMP_BASE="${ROOT_DIR}/.tmp"

mkdir -p "${TMP_BASE}" "${ROOT_DIR}/results"
export TMPDIR="${TMP_BASE}" TMP="${TMP_BASE}" TEMP="${TMP_BASE}"
export NUMBA_CACHE_DIR="${TMP_BASE}/numba_cache"
# Force PyTorch to use CUDA runtime libraries from the conda env first.
# This avoids mixing conda libcublas.so.11 with system libcublasLt.so.11.
export LD_LIBRARY_PATH="${ENV_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
mkdir -p "${NUMBA_CACHE_DIR}"

cd "${ROOT_DIR}"

echo "=============================="
echo "scPT V2 local h5ad evaluation"
echo "Python: ${PYTHON_BIN}"
echo "GPU: ${GPU_ID}"
echo "Data: ${ROOT_DIR}/infer_data"
echo "LD_LIBRARY_PATH first: ${ENV_PREFIX}/lib"
echo "=============================="

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" "${ROOT_DIR}/infer_scpt_v2.py" "$@"

echo "=============================="
echo "Evaluation DONE"
echo "Results: ${ROOT_DIR}/results"
echo "=============================="
