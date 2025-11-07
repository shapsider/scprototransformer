import scanpy as sc
import anndata as ad
import multimil as mtm
import numpy as np
import scvi
import time
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


data_path = "./HLCA_genept_light.h5ad"
adata = sc.read_h5ad(data_path)
print(adata)

sample_key = "sample"
samples = np.unique(adata.obs[sample_key])
n_samples = len(samples)
query_proportion = 0.2

rng = np.random.default_rng(0)
query_samples = rng.choice(samples, int(n_samples * query_proportion), replace=False)
query = adata[adata.obs[sample_key].isin(query_samples)].copy()
adata = adata[~adata.obs[sample_key].isin(query_samples)].copy()
query.obs["ref"] = "query"
adata.obs["ref"] = "reference"

classification_keys = ["disease"]
z_dim = 384
categorical_covariate_keys = classification_keys + [sample_key]
idx = adata.obs[sample_key].sort_values().index
adata = adata[idx].copy()
idx = query.obs[sample_key].sort_values().index
query = query[idx].copy()

mtm.model.MILClassifier.setup_anndata(
    adata,
    categorical_covariate_keys=categorical_covariate_keys,
)

mil = mtm.model.MILClassifier(
    adata,
    classification=classification_keys,
    z_dim=z_dim,
    sample_key=sample_key,
    class_loss_coef=0.1,
)

start_time = time.time()
mil.train(lr=1e-3)
end_time = time.time()
run_time = end_time - start_time
print(f"run time: {run_time}s")

mil.plot_losses()
plt.savefig("./figures/mil.pdf")

mil.get_model_output()
print(adata)
sc.pl.umap(adata, color=["cell type", "disease", "cell_attn"], ncols=1, frameon=False, vmax="p99", save=f"_attn.pdf")