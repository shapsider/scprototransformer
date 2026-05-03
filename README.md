## scProtoTransformer: Scalable Reference Mapping Across Molecules, Cells and Donors

Using our molecular embedding, we establish a reference mapping from molecules to cells to donors resolution.

All data and pre-trained weights used for demonstration are available on Google Drive: https://drive.google.com/drive/folders/1zh8E2i4dFrmSq0_U0VkKzyxpI-wQS5de?usp=sharing

## Environment Setup

```bash
bash ./scProtoTransformer/model/FoudationModel/install_env_scPT.sh
```

The environment is created at:

```bash
/data/miniconda3/envs/scPT
```

We begin with the most fundamental molecular embeddings and proceed sequentially through the gene-level tasks `gene_task`, cell-level task `cell` and donor-level task `donor`. 

We also provide a pre-trained scProtoTransformer, which allows for direct inference on unseen data. For specific usage instructions, please refer to: `./FoudationModel/README.md`.
