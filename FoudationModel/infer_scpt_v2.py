import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "7")

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn.functional as F

from scPT.scPT_model import scPT_model as create_model


ROOT = Path(__file__).resolve().parent
DEFAULT_PROJECT_DIR = ROOT / "scPT_v2"
DEFAULT_MERGE_MAP = DEFAULT_PROJECT_DIR / "merge_map_original_to_merged.csv"
DEFAULT_DATA_DIR = ROOT / "data"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate the packaged scPT V2 model on local h5ad files."
    )
    parser.add_argument(
        "--input-h5ad",
        nargs="*",
        default=[],
        help="Specific h5ad files. If omitted, all *.h5ad files from --data-dir are used.",
    )
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="Directory of h5ad files.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label-col", default="predictions_unconstrained")
    parser.add_argument("--max-cells-per-file", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--project-dir", default=str(DEFAULT_PROJECT_DIR))
    parser.add_argument("--merge-map", default=str(DEFAULT_MERGE_MAP))
    parser.add_argument("--output-dir", default=str(ROOT / "results"))
    parser.add_argument("--embed-dim", type=int, default=384)
    parser.add_argument("--depth", type=int, default=16)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--max-umap-cells", type=int, default=80000)
    parser.add_argument("--skip-umap", action="store_true")
    return parser.parse_args()


def to_dense_float32(x_mat):
    if sp.issparse(x_mat):
        x_mat = x_mat.toarray()
    return np.asarray(x_mat, dtype=np.float32)


def load_classes(project_dir):
    label_csv = project_dir / "merged_label_dictionary.csv"
    df = pd.read_csv(label_csv)
    if {"idx", "merged_class"}.issubset(df.columns):
        df = df.sort_values("idx")
        return df["merged_class"].astype(str).tolist()
    raise ValueError(f"Unsupported label dictionary format: {label_csv}")


def load_merge_map(path):
    df = pd.read_csv(path)
    if not {"original", "merged"}.issubset(df.columns):
        raise ValueError(f"Unsupported merge map format: {path}")
    return dict(zip(df["original"].astype(str), df["merged"].astype(str)))


def resolve_files(args):
    if args.input_h5ad:
        files = [Path(p) for p in args.input_h5ad]
    else:
        files = sorted(Path(args.data_dir).glob("*.h5ad"))
    return [p for p in files if p.exists()]


def load_model(args, device):
    project_dir = Path(args.project_dir)
    mask = np.load(project_dir / "mask.npy")
    gene_embeddings = np.load(project_dir / "gene_embeddings_aligned.npy")
    with open(project_dir / "selected_genes.txt") as f:
        selected_genes = [line.strip() for line in f if line.strip()]

    classes = load_classes(project_dir)
    model = create_model(
        num_classes=len(classes),
        num_genes=len(selected_genes),
        mask=mask,
        gene_embeddings=gene_embeddings,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        has_logits=False,
    ).to(device)
    state = torch.load(project_dir / "best_model.pth", map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, selected_genes, classes


def infer_matrix(model, x_mat, batch_size, device):
    preds = []
    probs = []
    n_cells = x_mat.shape[0]
    with torch.no_grad():
        for start in range(0, n_cells, batch_size):
            end = min(start + batch_size, n_cells)
            xb = torch.from_numpy(to_dense_float32(x_mat[start:end])).to(device)
            _, logits, _ = model(xb, return_attn=False)
            prob = F.softmax(logits, dim=-1)
            preds.append(torch.argmax(prob, dim=-1).cpu().numpy())
            probs.append(torch.max(prob, dim=-1)[0].cpu().numpy())
            del xb, logits, prob
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return np.concatenate(preds), np.concatenate(probs)


def save_umap(result_adata, output_dir, max_cells, seed, prefix):
    import scanpy as sc
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if result_adata.n_obs < 3 or result_adata.n_vars < 2:
        print("Skip UMAP: not enough cells or genes.", flush=True)
        return None

    if max_cells > 0 and result_adata.n_obs > max_cells:
        adata_umap = result_adata[:max_cells].copy()
    else:
        adata_umap = result_adata.copy()

    if sp.issparse(adata_umap.X):
        adata_umap.X = adata_umap.X.astype(np.float32)
    else:
        adata_umap.X = np.asarray(adata_umap.X, dtype=np.float32)

    n_comps = min(30, adata_umap.n_vars - 1, adata_umap.n_obs - 1)
    if n_comps < 2:
        print("Skip UMAP: PCA dimension would be too small.", flush=True)
        return None

    print("  -> Running PCA...", flush=True)
    sc.pp.pca(adata_umap, n_comps=n_comps)
    print("  -> Computing neighbors...", flush=True)
    sc.pp.neighbors(adata_umap)
    print("  -> Running UMAP...", flush=True)
    sc.tl.umap(adata_umap, random_state=seed)

    print("  -> Plotting UMAP...", flush=True)
    fig, axes = plt.subplots(1, 3, figsize=(30, 8))
    sc.pl.umap(
        adata_umap,
        color="gt_label_merged",
        ax=axes[0],
        show=False,
        title="GT label (merged)",
        legend_loc="on data",
        legend_fontsize=5,
    )
    sc.pl.umap(
        adata_umap,
        color="pred_label_merged",
        ax=axes[1],
        show=False,
        title="Pred label (merged)",
        legend_loc="on data",
        legend_fontsize=5,
    )
    sc.pl.umap(
        adata_umap,
        color="correct",
        ax=axes[2],
        show=False,
        title="Correct on merged labels",
        palette={"True": "#2ecc71", "False": "#e74c3c", "NA": "#95a5a6"},
    )
    plt.tight_layout()
    png_path = output_dir / f"{prefix}_umap_gt_vs_pred.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    umap_h5ad_path = output_dir / f"{prefix}_umap_subset.h5ad"
    adata_umap.write_h5ad(umap_h5ad_path)
    print(f"UMAP saved: {png_path}", flush=True)
    print(f"UMAP AnnData saved: {umap_h5ad_path}", flush=True)
    return png_path


def main():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)
    print(f"Device: {device}", flush=True)

    import anndata as ad
    from sklearn.metrics import accuracy_score, classification_report, f1_score

    model, selected_genes, classes = load_model(args, device)
    label_to_idx = {label: i for i, label in enumerate(classes)}
    merge_map = load_merge_map(Path(args.merge_map))
    files = resolve_files(args)
    if not files:
        print(f"No h5ad files found under: {args.data_dir}", flush=True)
        sys.exit(1)

    output_dir = Path(args.output_dir)
    result_h5ad_dir = output_dir / "result_h5ad_per_file"
    umap_dir = output_dir / "umap_per_file"
    output_dir.mkdir(parents=True, exist_ok=True)
    result_h5ad_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_umap:
        umap_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loaded model: {len(classes)} merged classes, {len(selected_genes)} selected genes", flush=True)
    print(f"Files to evaluate: {len(files)}", flush=True)
    print(f"Input h5ad dir: {Path(args.data_dir).resolve()}", flush=True)
    print(f"Output dir: {output_dir}", flush=True)

    result_adatas = []
    prediction_rows = []
    y_true = []
    y_pred = []

    for file_idx, h5ad_path in enumerate(files, 1):
        print(f"[{file_idx}/{len(files)}] {h5ad_path.name}", flush=True)
        adata = ad.read_h5ad(h5ad_path)
        if args.label_col not in adata.obs.columns:
            print(f"  skip: missing obs['{args.label_col}']", flush=True)
            continue

        var_index = pd.Index(adata.var_names.astype(str))
        gene_pos = var_index.get_indexer(selected_genes)
        if np.any(gene_pos < 0):
            print(f"  skip: missing {int((gene_pos < 0).sum())} selected genes", flush=True)
            continue

        chosen = np.arange(adata.n_obs)
        if args.max_cells_per_file > 0 and adata.n_obs > args.max_cells_per_file:
            chosen = np.arange(args.max_cells_per_file)

        input_adata = adata[chosen].copy()
        input_adata.obs["source_file"] = h5ad_path.name

        x_model = input_adata[:, gene_pos].X
        pred_idx, conf = infer_matrix(model, x_model, args.batch_size, device)
        pred_merged = np.array([classes[int(i)] for i in pred_idx])

        gt_original = input_adata.obs[args.label_col].astype(str).to_numpy()
        gt_merged = np.array([merge_map.get(x, x) for x in gt_original])
        valid = np.array([x in label_to_idx for x in gt_merged])
        gt_merged_display = np.array([x if ok else "Unknown" for x, ok in zip(gt_merged, valid)])
        correct = np.full(input_adata.n_obs, "NA", dtype=object)
        correct[valid] = (gt_merged[valid] == pred_merged[valid]).astype(str)

        y_true.extend(gt_merged[valid].tolist())
        y_pred.extend(pred_merged[valid].tolist())

        result = input_adata[:, gene_pos].copy()
        result.obs["source_file"] = h5ad_path.name
        result.obs["gt_label_original"] = gt_original
        result.obs["gt_label_merged"] = pd.Categorical(gt_merged_display, categories=classes + ["Unknown"])
        result.obs["pred_label_merged"] = pd.Categorical(pred_merged, categories=classes)
        result.obs["pred_confidence"] = conf
        result.obs["correct"] = pd.Categorical(correct, categories=["True", "False", "NA"])
        result.uns["model"] = "scPT_v2"
        result.uns["weight"] = str(Path(args.project_dir) / "best_model.pth")
        result.uns["source_h5ad"] = str(h5ad_path)

        result_path = result_h5ad_dir / f"{h5ad_path.stem}.result.h5ad"
        result.write_h5ad(result_path)
        result_adatas.append(result)

        for cell_id, gt_o, gt_m, pred_m, prob, ok in zip(
            result.obs_names.astype(str),
            gt_original,
            gt_merged,
            pred_merged,
            conf,
            correct,
        ):
            prediction_rows.append(
                {
                    "source_file": h5ad_path.name,
                    "cell_id": cell_id,
                    "gt_label_original": gt_o,
                    "gt_label_merged": gt_m if gt_m in label_to_idx else "Unknown",
                    "pred_label_merged": pred_m,
                    "pred_confidence": float(prob),
                    "correct": ok,
                }
            )

        if valid.any():
            file_acc = np.mean(gt_merged[valid] == pred_merged[valid])
            print(
                f"  cells={input_adata.n_obs} valid={int(valid.sum())} acc={file_acc:.4f}",
                flush=True,
            )
        else:
            print(f"  cells={input_adata.n_obs} no valid merged GT labels", flush=True)
        print(f"  saved result h5ad:  {result_path}", flush=True)

        if not args.skip_umap:
            print(f"  -> Generating UMAP for {h5ad_path.name}...", flush=True)
            save_umap(result, umap_dir, args.max_umap_cells, args.seed, h5ad_path.stem)

    if not result_adatas:
        print("No valid h5ad files evaluated.", flush=True)
        sys.exit(1)

    result_adata = ad.concat(result_adatas, join="inner", index_unique="-")
    result_adata_path = output_dir / "result_adata.h5ad"
    result_adata.write_h5ad(result_adata_path)
    print(f"Combined result AnnData saved: {result_adata_path}", flush=True)

    predictions_path = output_dir / "predictions.csv"
    pd.DataFrame(prediction_rows).to_csv(predictions_path, index=False)
    print(f"Predictions saved: {predictions_path}", flush=True)

    if y_true:
        acc = accuracy_score(y_true, y_pred)
        f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        summary = pd.DataFrame(
            [
                {
                    "n_files": len(result_adatas),
                    "n_cells_predicted": result_adata.n_obs,
                    "n_cells_with_valid_merged_gt": len(y_true),
                    "accuracy_merged": acc,
                    "f1_weighted_merged": f1_weighted,
                }
            ]
        )
        summary_path = output_dir / "summary.csv"
        report_path = output_dir / "classification_report_merged.csv"
        summary.to_csv(summary_path, index=False)
        pd.DataFrame(
            classification_report(y_true, y_pred, zero_division=0, output_dict=True)
        ).transpose().to_csv(report_path)

        print("\nMerged-label evaluation", flush=True)
        print(f"Accuracy:      {acc:.4f}", flush=True)
        print(f"F1 weighted:   {f1_weighted:.4f}", flush=True)
        print(f"Summary saved: {summary_path}", flush=True)
        print(f"Report saved:  {report_path}", flush=True)
    else:
        print("No valid merged GT labels found; metrics were not computed.", flush=True)

if __name__ == "__main__":
    main()
