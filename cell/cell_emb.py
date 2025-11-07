import scanpy as sc
import pandas as pd
import seaborn as sns
sns.set_style("whitegrid")
import pandas as pd
import numpy as np
import anndata as ad
import sys
import gc  # 添加垃圾回收模块


adata_gene = sc.read("./gene_emb_scProtoTransformer.h5ad")
adata_gene.obsm['X_emb'] = adata_gene.X
print(adata_gene.obs_names)

GPT_3_5_gene_embeddings = dict(zip(adata_gene.obs.index, adata_gene.obsm["X_emb"]))
print(GPT_3_5_gene_embeddings['LINC02844'].shape) # (384,)

sampled_adata = sc.read_h5ad("./sample_aorta_data_updated.h5ad")
# sampled_adata = sc.pp.subsample(sampled_adata, fraction=0.1, copy=True)
print(sampled_adata)
print(type(sampled_adata.X))
print(sampled_adata.X.max())
print(sampled_adata.X.min())
sc.set_figure_params(dpi=80)
sc.pl.umap(sampled_adata, color=["celltype", "patient"], save=f"_sc_count.pdf")

def genept_w_embedding():
    gene_names = list(sampled_adata.var.index)
    count_missing = 0
    EMBED_DIM = 384
    lookup_embed = np.zeros(shape=(len(gene_names), EMBED_DIM))
    for i, gene in enumerate(gene_names):
        if gene in GPT_3_5_gene_embeddings:
            lookup_embed[i,:] = GPT_3_5_gene_embeddings[gene]
        else:
            count_missing += 1
    print(f"Unable to match {count_missing} out of {len(gene_names)} genes in the GenePT-w embedding")

    # 分批次计算加权embedding
    batch_size = 500  # 每批处理的细胞数
    n_cells = sampled_adata.shape[0]
    genePT_w_emebed = np.zeros((n_cells, EMBED_DIM))

    for i in range(0, n_cells, batch_size):
        end_idx = min(i + batch_size, n_cells)
        batch_cells = sampled_adata[i:end_idx].X
        batch_embed = np.dot(batch_cells, lookup_embed) / len(gene_names)
        genePT_w_emebed[i:end_idx] = batch_embed
        
        if (i // batch_size) % 5 == 0:
            print(f"Processed {end_idx}/{n_cells} cells")
        
        # 强制垃圾回收，释放内存
        gc.collect()

    # 归一化
    genePT_w_emebed = genePT_w_emebed / np.linalg.norm(
        genePT_w_emebed, axis=1, keepdims=True
    )
    sampled_adata.obsm['genept'] = genePT_w_emebed

    print("UMAP")
    sc.pp.neighbors(sampled_adata, 
                    use_rep="genept", metric="cosine")
    sc.tl.umap(sampled_adata)

    sc.set_figure_params(dpi=80)
    sc.pl.umap(sampled_adata, color=["celltype", "patient"], save=f"_sc_genept-w.pdf")


genept_w_embedding()