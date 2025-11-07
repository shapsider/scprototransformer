import scanpy as sc
import pandas as pd
import seaborn as sns
sns.set_style("whitegrid")
import pandas as pd
import numpy as np 
import scipy.stats as stats
from collections import Counter
import matplotlib.pyplot as plt
import umap
import matplotlib
import mygene
import pickle
import sklearn
import random
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from xgboost import XGBClassifier
# import sentence_transformers
plt.style.use('ggplot')
#plt.style.use('seaborn-v0_8-dark-palette')
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})
import matplotlib_inline
matplotlib_inline.backend_inline.set_matplotlib_formats('retina')

adata_gene = sc.read("./gene_emb_scProtoTransformer.h5ad")
adata_gene.obsm['X_emb'] = adata_gene.X
print(adata_gene.obs_names)

mapping_df = pd.read_csv("./example_input_files/gene_info_table.csv")
id_to_symbol = dict(zip(mapping_df['ensembl_id'], mapping_df['gene_name']))

GPT_3_5_gene_embeddings = dict(zip(adata_gene.obs.index, adata_gene.obsm["X_emb"]))
print(GPT_3_5_gene_embeddings['LINC02844'].shape)

def tf_range():
    with open('./example_input_files/gene_classification/tf_regulatory_range/tf_regulatory_range.pickle', 'rb') as f:
        data = pickle.load(f)

    long_range_tf_gene = data['long_range']
    long_range_tf_gene = [id_to_symbol.get(id_, f"{id_}_UNKNOWN") for id_ in long_range_tf_gene]
    short_range_tf_gene = data['short_range']
    short_range_tf_gene = [id_to_symbol.get(id_, f"{id_}_UNKNOWN") for id_ in short_range_tf_gene]
    print(long_range_tf_gene)
    print(short_range_tf_gene)

    x_long_range_tf = [GPT_3_5_gene_embeddings[x] for x in long_range_tf_gene \
                if x in GPT_3_5_gene_embeddings]
    x_short_range_tf =  [GPT_3_5_gene_embeddings[x] for x in short_range_tf_gene\
                if x in GPT_3_5_gene_embeddings]

    # np.random.seed(2023)
    # random.seed(2023)

    X_array = np.concatenate((x_long_range_tf,x_short_range_tf))
    y_array =  np.concatenate((np.repeat(1,len(x_long_range_tf)),np.repeat(0,len(x_short_range_tf))))

    cv = StratifiedKFold(n_splits=5)

    roc_auc_logistic = []
    roc_auc_rf = []

    tpr_logistic = []
    fpr_logistic = []
    tpr_rf = []
    fpr_rf = []

    for train_index, test_index in cv.split(X_array, y_array):
        X_train, X_test = X_array[train_index], X_array[test_index]
        y_train, y_test = y_array[train_index], y_array[test_index]

        # Logistic Regression
        logistic_model = LogisticRegression()
        logistic_model.fit(X_train, y_train)
        y_score_logistic = logistic_model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_score_logistic)
        roc_auc = auc(fpr, tpr)
        roc_auc_logistic.append(roc_auc)
        tpr_logistic.append(tpr)
        fpr_logistic.append(fpr)

        # Random Forest
        random_forest_model = RandomForestClassifier()
        random_forest_model.fit(X_train, y_train)
        y_score_rf = random_forest_model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_score_rf)
        roc_auc = auc(fpr, tpr)
        roc_auc_rf.append(roc_auc)
        tpr_rf.append(tpr)
        fpr_rf.append(fpr)

    print(f"Logistic Regression ROC AUC: {np.mean(roc_auc_logistic):.3f} +/- {np.std(roc_auc_logistic):.3f}")
    print(f"Random Forest ROC AUC: {np.mean(roc_auc_rf):.3f} +/- {np.std(roc_auc_rf):.3f}")

def dosage_sensitivity():
    with open('./example_input_files/gene_classification/dosage_sensitive_tfs/dosage_sensitivity_TFs.pickle', 'rb') as f:
        data = pickle.load(f)

    sensitive = data["Dosage-sensitive TFs"]
    insensitive = data["Dosage-insensitive TFs"]

    sensitive_gene_name = [id_to_symbol.get(id_, f"{id_}_UNKNOWN") for id_ in sensitive]
    in_sensitive_gene_name = [id_to_symbol.get(id_, f"{id_}_UNKNOWN") for id_ in insensitive]

    x_sensitive = [GPT_3_5_gene_embeddings[name] for name in sensitive_gene_name\
                if name in GPT_3_5_gene_embeddings]
    x_insensitive = [GPT_3_5_gene_embeddings[name] for name in in_sensitive_gene_name \
                    if name in GPT_3_5_gene_embeddings]
    x_dosage = x_sensitive.copy()
    x_dosage.extend(x_insensitive)
    y_dosage = np.concatenate((np.repeat(1,len(x_sensitive)),np.repeat(0,len(x_insensitive))))

    X_array = np.array(x_dosage)
    y_array = np.array(y_dosage)

    cv = StratifiedKFold(n_splits=5)

    roc_auc_logistic = []
    roc_auc_rf = []

    tpr_logistic = []
    fpr_logistic = []
    tpr_rf = []
    fpr_rf = []

    for train_index, test_index in cv.split(X_array, y_array):
        X_train, X_test = X_array[train_index], X_array[test_index]
        y_train, y_test = y_array[train_index], y_array[test_index]

        # Logistic Regression
        logistic_model = LogisticRegression()
        logistic_model.fit(X_train, y_train)
        y_score_logistic = logistic_model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_score_logistic)
        roc_auc = auc(fpr, tpr)
        roc_auc_logistic.append(roc_auc)
        tpr_logistic.append(tpr)
        fpr_logistic.append(fpr)

        # Random Forest
        random_forest_model = RandomForestClassifier()
        random_forest_model.fit(X_train, y_train)
        y_score_rf = random_forest_model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_score_rf)
        roc_auc = auc(fpr, tpr)
        roc_auc_rf.append(roc_auc)
        tpr_rf.append(tpr)
        fpr_rf.append(fpr)

    # Print ROC AUC scores
    print(f"Logistic Regression ROC AUC: {np.mean(roc_auc_logistic):.3f} +/- {np.std(roc_auc_logistic):.3f}")
    print(f"Random Forest ROC AUC: {np.mean(roc_auc_rf):.3f} +/- {np.std(roc_auc_rf):.3f}")


def bivalent_vs_no_methyl():
    with open('./example_input_files/gene_classification/bivalent_promoters/bivalent_vs_no_methyl.pickle', 'rb') as f:
        data = pickle.load(f)

    bivalent_gene_labels = data['bivalent']
    no_methylation_gene_labels = data['no_methylation']

    bivalent_gene_name = [id_to_symbol.get(id_, f"{id_}_UNKNOWN") for id_ in bivalent_gene_labels]
    no_methylation_gene_name = [id_to_symbol.get(id_, f"{id_}_UNKNOWN") for id_ in no_methylation_gene_labels]


    x_bivalent = [GPT_3_5_gene_embeddings[name] for name in bivalent_gene_name\
                if name in GPT_3_5_gene_embeddings]
    x_no_methylation = [GPT_3_5_gene_embeddings[name] for name in no_methylation_gene_name \
                    if name in GPT_3_5_gene_embeddings]

    X_array = np.concatenate((x_bivalent,x_no_methylation))
    y_array =  np.concatenate((np.repeat(1,len(x_bivalent)),np.repeat(0,len(x_no_methylation))))

    cv = StratifiedKFold(n_splits=5)

    roc_auc_logistic = []
    roc_auc_rf = []

    tpr_logistic = []
    fpr_logistic = []
    tpr_rf = []
    fpr_rf = []

    for train_index, test_index in cv.split(X_array, y_array):
        X_train, X_test = X_array[train_index], X_array[test_index]
        y_train, y_test = y_array[train_index], y_array[test_index]

        # Logistic Regression
        logistic_model = LogisticRegression()
        logistic_model.fit(X_train, y_train)
        y_score_logistic = logistic_model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_score_logistic)
        roc_auc = auc(fpr, tpr)
        roc_auc_logistic.append(roc_auc)
        tpr_logistic.append(tpr)
        fpr_logistic.append(fpr)

        # Random Forest
        random_forest_model = RandomForestClassifier()
        random_forest_model.fit(X_train, y_train)
        y_score_rf = random_forest_model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_score_rf)
        roc_auc = auc(fpr, tpr)
        roc_auc_rf.append(roc_auc)
        tpr_rf.append(tpr)
        fpr_rf.append(fpr)

    # Print ROC AUC scores
    print(f"Logistic Regression ROC AUC: {np.mean(roc_auc_logistic):.3f} +/- {np.std(roc_auc_logistic):.3f}")
    print(f"Random Forest ROC AUC: {np.mean(roc_auc_rf):.3f} +/- {np.std(roc_auc_rf):.3f}")


def bivalent_vs_lys4():
    with open('./example_input_files/gene_classification/bivalent_promoters/bivalent_vs_lys4_only.pickle', 'rb') as f:
        data = pickle.load(f)

    bivalent_gene_labels = data['bivalent']
    lysine_gene_labels = data['lys4_only']

    bivalent_gene_name = [id_to_symbol.get(id_, f"{id_}_UNKNOWN") for id_ in bivalent_gene_labels]
    lysine_gene_name = [id_to_symbol.get(id_, f"{id_}_UNKNOWN") for id_ in lysine_gene_labels]


    x_bivalent = [GPT_3_5_gene_embeddings[name] for name in bivalent_gene_name\
                if name in GPT_3_5_gene_embeddings]
    x_lysine = [GPT_3_5_gene_embeddings[name] for name in lysine_gene_name \
                    if name in GPT_3_5_gene_embeddings]

    X_array = np.concatenate((x_bivalent,x_lysine))
    y_array =  np.concatenate((np.repeat(1,len(x_bivalent)),np.repeat(0,len(x_lysine))))

    cv = StratifiedKFold(n_splits=5)

    roc_auc_logistic = []
    roc_auc_rf = []

    tpr_logistic = []
    fpr_logistic = []
    tpr_rf = []
    fpr_rf = []

    for train_index, test_index in cv.split(X_array, y_array):
        X_train, X_test = X_array[train_index], X_array[test_index]
        y_train, y_test = y_array[train_index], y_array[test_index]

        # Logistic Regression
        logistic_model = LogisticRegression()
        logistic_model.fit(X_train, y_train)
        y_score_logistic = logistic_model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_score_logistic)
        roc_auc = auc(fpr, tpr)
        roc_auc_logistic.append(roc_auc)
        tpr_logistic.append(tpr)
        fpr_logistic.append(fpr)

        # Random Forest
        random_forest_model = RandomForestClassifier()
        random_forest_model.fit(X_train, y_train)
        y_score_rf = random_forest_model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_score_rf)
        roc_auc = auc(fpr, tpr)
        roc_auc_rf.append(roc_auc)
        tpr_rf.append(tpr)
        fpr_rf.append(fpr)

    # Print ROC AUC scores
    print(f"Logistic Regression ROC AUC: {np.mean(roc_auc_logistic):.3f} +/- {np.std(roc_auc_logistic):.3f}")
    print(f"Random Forest ROC AUC: {np.mean(roc_auc_rf):.3f} +/- {np.std(roc_auc_rf):.3f}")

print("++++++++++++++++")
tf_range()
print("++++++++++++++++")
dosage_sensitivity()
print("++++++++++++++++")
bivalent_vs_no_methyl()
print("++++++++++++++++")
bivalent_vs_lys4()
