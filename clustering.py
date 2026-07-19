"""
IPO Research — Part 1: Clustering (Steps 1-6 of CLUSTERING_PLAN.md)

  1. Finalize feature matrix (winsorize outlier-dominated cols, re-scale, drop industry)
  2. Determine optimal k (elbow / silhouette / Calinski-Harabasz / Davies-Bouldin)
  3. PCA 2D projection (visualization)
  4. Fit & compare K-Means vs Agglomerative (Ward)
  5. Profile clusters (feature means + industry mix)
  6. Link clusters to IPO performance (returns + Kruskal-Wallis)

Outputs:
  ipo_features_for_clustering.csv
  cluster_k_selection.png
  pca_projection.png
  dendrogram.png
  cluster_profiles.xlsx
  cluster_performance.xlsx
  ipo_clustered.xlsx          (preprocessed table + cluster labels)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import kruskal
from scipy.stats.mstats import winsorize
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import RobustScaler

RANDOM_STATE = 42

CLUSTER_FEATURES = [
    "Company_Age", "Log_Issue_Price", "Range_Width_pct", "Days_to_List",
    "Revenue_CAGR", "PAT_CAGR", "Avg_Operating_Margin", "Avg_ROE_pct",
    "Avg_ROCE_pct", "Avg_DE_pct", "Avg_Interest_Coverage",
    "Avg_Assets_Turnover", "Log_Latest_Revenue",
]

RETURN_COLS = [
    "Return_1d", "Return_7d", "Return_15d", "Return_21d",
    "Return_30d", "Return_45d", "Return_60d", "PostIPO_price_std",
]


def winsorize_col(series, limits=(0.05, 0.05)):
    arr = winsorize(series.dropna().astype(float), limits=limits)
    result = series.copy().astype(float)
    result[series.notna()] = np.asarray(arr)
    return result


# ── Step 1: finalize feature matrix ──────────────────────────────────────────

print("Step 1 — Finalizing feature matrix...")
pp = pd.read_excel("ipo_preprocessed.xlsx")

feat = pp[["COMPANY NAME", "Symbol"] + CLUSTER_FEATURES].copy()

# Winsorize the two outlier-dominated columns BEFORE scaling
for col in ["Range_Width_pct", "Days_to_List"]:
    before = (feat[col].min(), feat[col].max())
    feat[col] = winsorize_col(feat[col])
    after = (feat[col].min(), feat[col].max())
    print(f"  Winsorized {col}: [{before[0]:.1f}, {before[1]:.1f}] → [{after[0]:.1f}, {after[1]:.1f}]")

# Re-scale with RobustScaler
scaler = RobustScaler()
X = scaler.fit_transform(feat[CLUSTER_FEATURES])
X_df = pd.DataFrame(X, columns=CLUSTER_FEATURES)
X_df.insert(0, "Symbol", feat["Symbol"].values)
X_df.insert(0, "COMPANY NAME", feat["COMPANY NAME"].values)
X_df.to_csv("ipo_features_for_clustering.csv", index=False)

# Confirm variances are now comparable
variances = X_df[CLUSTER_FEATURES].var().sort_values()
print(f"  Post-scaling variance range: {variances.min():.2f} – {variances.max():.2f}")
print(f"  (was up to ~573 before winsorizing Days_to_List)")

# ── Step 2: determine optimal k ──────────────────────────────────────────────

print("\nStep 2 — Determining optimal k (2..10)...")
k_range = range(2, 11)
inertias, sils, chs, dbs = [], [], [], []
for k in k_range:
    km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
    labels = km.fit_predict(X)
    inertias.append(km.inertia_)
    sils.append(silhouette_score(X, labels))
    chs.append(calinski_harabasz_score(X, labels))
    dbs.append(davies_bouldin_score(X, labels))
    print(f"  k={k}: silhouette={sils[-1]:.3f}, CH={chs[-1]:.0f}, DB={dbs[-1]:.3f}")

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
axes[0, 0].plot(list(k_range), inertias, "o-"); axes[0, 0].set_title("Elbow (Inertia)"); axes[0, 0].set_xlabel("k")
axes[0, 1].plot(list(k_range), sils, "o-", color="green"); axes[0, 1].set_title("Silhouette (higher=better)"); axes[0, 1].set_xlabel("k")
axes[1, 0].plot(list(k_range), chs, "o-", color="purple"); axes[1, 0].set_title("Calinski-Harabasz (higher=better)"); axes[1, 0].set_xlabel("k")
axes[1, 1].plot(list(k_range), dbs, "o-", color="red"); axes[1, 1].set_title("Davies-Bouldin (lower=better)"); axes[1, 1].set_xlabel("k")
plt.tight_layout(); plt.savefig("cluster_k_selection.png", dpi=120); plt.close()

best_k = list(k_range)[int(np.argmax(sils))]
print(f"  Best k by silhouette: {best_k}")

# ── Step 3: PCA projection ───────────────────────────────────────────────────

print("\nStep 3 — PCA 2D projection...")
pca = PCA(n_components=2, random_state=RANDOM_STATE)
pcs = pca.fit_transform(X)
var_explained = pca.explained_variance_ratio_
print(f"  PC1 explains {var_explained[0]*100:.1f}%, PC2 explains {var_explained[1]*100:.1f}%")

# ── Step 4: fit & compare K-Means vs Agglomerative ──────────────────────────

print(f"\nStep 4 — Fitting K-Means & Agglomerative at k={best_k}...")
km = KMeans(n_clusters=best_k, n_init=10, random_state=RANDOM_STATE)
km_labels = km.fit_predict(X)
km_sil = silhouette_score(X, km_labels)

agg = AgglomerativeClustering(n_clusters=best_k, linkage="ward")
agg_labels = agg.fit_predict(X)
agg_sil = silhouette_score(X, agg_labels)

print(f"  K-Means silhouette:      {km_sil:.3f}")
print(f"  Agglomerative silhouette: {agg_sil:.3f}")

if km_sil >= agg_sil:
    primary_name, primary_labels = "K-Means", km_labels
else:
    primary_name, primary_labels = "Agglomerative", agg_labels
print(f"  Primary model chosen: {primary_name}")

# Dendrogram
plt.figure(figsize=(13, 6))
Z = linkage(X, method="ward")
dendrogram(Z, truncate_mode="level", p=5, no_labels=True)
plt.title("Hierarchical Clustering Dendrogram (Ward)")
plt.xlabel("Companies"); plt.ylabel("Distance")
plt.tight_layout(); plt.savefig("dendrogram.png", dpi=120); plt.close()

# PCA scatter colored by primary clusters
plt.figure(figsize=(9, 7))
scatter = plt.scatter(pcs[:, 0], pcs[:, 1], c=primary_labels, cmap="tab10", s=40, alpha=0.8)
plt.colorbar(scatter, label="Cluster")
plt.title(f"PCA Projection — {primary_name} (k={best_k})\nPC1 {var_explained[0]*100:.0f}% / PC2 {var_explained[1]*100:.0f}%")
plt.xlabel("PC1"); plt.ylabel("PC2")
plt.tight_layout(); plt.savefig("pca_projection.png", dpi=120); plt.close()

# Attach labels to the full preprocessed table
pp_out = pp.copy()
pp_out["Cluster"] = primary_labels
pp_out["Cluster_KMeans"] = km_labels
pp_out["Cluster_Agglomerative"] = agg_labels
pp_out.to_excel("ipo_clustered.xlsx", index=False)

# ── Step 5: profile clusters ─────────────────────────────────────────────────

print("\nStep 5 — Profiling clusters...")
profile = pp_out.groupby("Cluster")[CLUSTER_FEATURES].mean().round(2)
profile["N_companies"] = pp_out.groupby("Cluster").size()
profile = profile[["N_companies"] + CLUSTER_FEATURES]
print(profile.to_string())

# Industry composition per cluster
ind_mix = pd.crosstab(pp_out["Cluster"], pp_out["Assigned Industry"])
ind_mix_pct = pd.crosstab(pp_out["Cluster"], pp_out["Assigned Industry"], normalize="index").round(3) * 100

with pd.ExcelWriter("cluster_profiles.xlsx") as writer:
    profile.to_excel(writer, sheet_name="Feature_Means")
    ind_mix.to_excel(writer, sheet_name="Industry_Counts")
    ind_mix_pct.to_excel(writer, sheet_name="Industry_Pct")

# ── Step 6: link clusters to IPO performance ────────────────────────────────

print("\nStep 6 — Linking clusters to IPO performance...")
targets = pd.read_csv("ipo_targets.csv")
merged = pp_out[["COMPANY NAME", "Cluster"]].merge(targets, on="COMPANY NAME", how="left")

perf = merged.groupby("Cluster")[RETURN_COLS].median().round(2)
perf["N_with_returns"] = merged.dropna(subset=["Return_1d"]).groupby("Cluster").size()
print("\n  Median returns per cluster:")
print(perf.to_string())

# Kruskal-Wallis test per return horizon
print("\n  Kruskal-Wallis (do clusters differ in returns?):")
kw_results = []
for col in RETURN_COLS:
    groups = [g[col].dropna().values for _, g in merged.groupby("Cluster")]
    groups = [g for g in groups if len(g) > 0]
    if len(groups) >= 2:
        stat, p = kruskal(*groups)
        sig = "YES" if p < 0.05 else "no"
        kw_results.append({"Metric": col, "H_statistic": round(stat, 2), "p_value": round(p, 4), "Significant(p<0.05)": sig})
        print(f"    {col:20s}: H={stat:6.2f}, p={p:.4f}  significant={sig}")

kw_df = pd.DataFrame(kw_results)
with pd.ExcelWriter("cluster_performance.xlsx") as writer:
    perf.to_excel(writer, sheet_name="Median_Returns")
    merged.groupby("Cluster")[RETURN_COLS].mean().round(2).to_excel(writer, sheet_name="Mean_Returns")
    kw_df.to_excel(writer, sheet_name="KruskalWallis", index=False)

print("\nSaved outputs:")
for f in ["ipo_features_for_clustering.csv", "cluster_k_selection.png", "pca_projection.png",
          "dendrogram.png", "cluster_profiles.xlsx", "cluster_performance.xlsx", "ipo_clustered.xlsx"]:
    print(f"  {f}")
print("\nDone.")
