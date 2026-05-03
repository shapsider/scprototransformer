# scPT V2 Foundation Model

This directory is a clean local h5ad evaluation package for the trained
`scPT_v2` model. By default it evaluates all `.h5ad` files under `data/`.

## Contents

- `infer_scpt_v2.py`: local h5ad evaluation entrypoint. It defaults to GPU 7.
- `run_infer_gpu7.sh`: wrapper that runs inference with `CUDA_VISIBLE_DEVICES=7`.
- `install_env_scPT.sh`: creates the `scPT` conda environment at `/data/miniconda3/envs/scPT`.
- `scPT/`: minimal model code needed for inference.
- `scPT_v2/`: packaged weights, model metadata, and label mappings.
- `data/`: input h5ad files to evaluate.
- `results/`: default output directory.

## Install Environment

```bash
bash /aaa/gelseywang/buddy1/lukatang/scDeepSeek/scProtoTransformer/model/FoudationModel/install_env_scPT.sh
```

The environment is created at:

```bash
/data/miniconda3/envs/scPT
```

This follows the original TOSICA setup pattern, but installs `scanpy` with pip
instead of conda and also installs `tensorboard` and `einops`:

```bash
conda create -p /data/miniconda3/envs/scPT python=3.8
conda install pytorch=1.7.1 torchvision=0.8.2 torchaudio=0.7.2 cudatoolkit=11.0 -c pytorch
pip install scanpy==1.9.8 tensorboard einops
```

The environment path is `/data/miniconda3/envs/scPT`. The PyTorch build uses
`cudatoolkit=11.0`, which matches A100 servers with NVIDIA driver `450.80.02`
and CUDA `11.0`.

Check CUDA after installation:

```bash
LD_LIBRARY_PATH=/data/miniconda3/envs/scPT/lib:${LD_LIBRARY_PATH:-} \
/data/miniconda3/envs/scPT/bin/python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")
PY
```

If importing torch reports a `libcublas.so.11` / `libcublasLt.so.11` symbol
error, it means CUDA libraries are being mixed between the conda environment and
system CUDA. Keep `/data/miniconda3/envs/scPT/lib` first in `LD_LIBRARY_PATH`;
`run_infer_gpu7.sh` already does this automatically.

## Run Inference on GPU 7

Put the h5ad files under:

```bash
/aaa/gelseywang/buddy1/lukatang/scDeepSeek/scProtoTransformer/model/FoudationModel/data/
```

Then run evaluation. The script reads all `.h5ad` files from `data/` in sorted
order, runs inference, saves result AnnData, computes Acc/F1 on merged labels,
and draws UMAP comparisons:

```bash
bash /aaa/gelseywang/buddy1/lukatang/scDeepSeek/scProtoTransformer/model/FoudationModel/run_infer_gpu7.sh \
  --batch-size 128
```

To limit cells per file deterministically, pass `--max-cells-per-file N`. The
default is `0`, which means all cells:

```bash
bash /aaa/gelseywang/buddy1/lukatang/scDeepSeek/scProtoTransformer/model/FoudationModel/run_infer_gpu7.sh \
  --max-cells-per-file 50000 \
  --batch-size 128
```

You can still override with specific h5ad files:

```bash
bash /aaa/gelseywang/buddy1/lukatang/scDeepSeek/scProtoTransformer/model/FoudationModel/run_infer_gpu7.sh \
  --input-h5ad /path/to/a.h5ad /path/to/b.h5ad \
  --batch-size 128
```

Outputs are written by default to:

```bash
/aaa/gelseywang/buddy1/lukatang/scDeepSeek/scProtoTransformer/model/FoudationModel/results/
```

Main outputs:

- `results/result_adata.h5ad`: combined result AnnData with selected genes.
- `results/result_h5ad_per_file/*.result.h5ad`: per-file result AnnData.
- `results/predictions.csv`: per-cell GT/pred/confidence table.
- `results/summary.csv`: merged-label Accuracy and weighted F1.
- `results/classification_report_merged.csv`: merged-label classification report.
- `results/umap_per_file/*.png`: UMAP comparisons of merged GT and pred labels per file.
- `results/umap_per_file/*.h5ad`: AnnData subsets used for each UMAP.

If the h5ad has `obs["predictions_unconstrained"]`, the script also computes
accuracy and weighted F1 after mapping original labels into the 55
merged classes using `scPT_v2/merge_map_original_to_merged.csv`.