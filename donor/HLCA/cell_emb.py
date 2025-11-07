import scanpy as sc
import pandas as pd
import seaborn as sns
sns.set_style("whitegrid")
import pandas as pd
import numpy as np
import anndata as ad
import sys
import gc  # 添加垃圾回收模块

sampled_adata = sc.read("./HLCA_pulmonary_fibrosis_light.h5ad")
adata_multimil_demo = sc.read("./hlca_tutorial.h5ad")
sampled_adata.obs["cell type"] = adata_multimil_demo.obs['ann_level_3_label_final']

sampled_adata = sc.pp.subsample(sampled_adata, fraction=0.1, copy=True)
sc.pp.normalize_total(sampled_adata)
sc.pp.log1p(sampled_adata)
sc.pp.highly_variable_genes(sampled_adata, n_top_genes=5000, subset=True)
del sampled_adata.var
sampled_adata.X = sampled_adata.X.toarray()
print(sampled_adata)
print(type(sampled_adata.X))
print(sampled_adata.X.max())
print(sampled_adata.X.min())

sampled_adata.write("./HLCA_pulmonary_fibrosis_light_hvg.h5ad")

# sys.exit()

sampled_adata = sc.read("./HLCA_pulmonary_fibrosis_light_hvg.h5ad")

adata_gene = sc.read("./gene_emb_scProtoTransformer.h5ad")
adata_gene.obsm['X_emb'] = adata_gene.X
print(adata_gene.obs_names)

GPT_3_5_gene_embeddings = dict(zip(adata_gene.obs.index, adata_gene.obsm["X_emb"]))

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
    batch_size = 1000  # 每批处理的细胞数
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
    sc.pl.umap(sampled_adata, color=["cell type", "disease"], save=f"_sc_genept-w.pdf")
    sampled_adata.write("./HLCA_genept.h5ad")

    new_adata = ad.AnnData(
        X=sampled_adata.obsm["genept"].copy(),  # 将 genept 设为新的 X
        obs=sampled_adata.obs.copy(),            # 复制原有 obs
        obsm=sampled_adata.obsm.copy()          # 复制原有 obsm（注意：此时 obsm["genept"] 会重复存在）
    )
    del new_adata.obsm["genept"]
    print(new_adata)
    print(new_adata.obs_names)
    new_adata.write("./HLCA_genept_light.h5ad")

genept_w_embedding()