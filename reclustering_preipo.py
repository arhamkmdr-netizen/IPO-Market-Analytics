"""
Re-clustering on PRE-IPO snapshot features (not 2022-2026 averages).

Uses each company's last-FY-before-listing fundamentals + company age AT IPO,
restricted to the ~230-company cohort with a clean pre-IPO snapshot.
Compares against the original averaged-feature clustering and profiles clusters
against listing returns AND listing year (market cycle).

Outputs:
  reclustering_k_selection.png
  reclustering_pca.png
  ipo_reclustered_preipo.xlsx
  recluster_profiles.xlsx
  recluster_performance.xlsx
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import kruskal
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import RobustScaler

RANDOM_STATE = 42

# Pre-IPO snapshot features (financials investors saw) + issue-structure features
CLUSTER_FEATURES = [
    "Company_Age_at_IPO", "Log_Issue_Price", "Range_Width_pct", "Days_to_List",
    "PreIPO_Revenue_Growth", "PreIPO_PAT_Growth", "PreIPO_Operating_Margin",
    "PreIPO_ROE_pct", "PreIPO_ROCE_pct", "PreIPO_DE_pct",
    "PreIPO_Interest_Coverage", "PreIPO_Assets_Turnover", "Log_PreIPO_Revenue",
]
RETURN_COLS = ["Return_1d", "Return_30d", "Return_60d", "PostIPO_price_std"]


def winsorize_clip(s, lo=0.05, hi=0.95):
    return s.clip(s.quantile(lo), s.quantile(hi))


# ── Assemble feature table ───────────────────────────────────────────────────
print("Assembling pre-IPO clustering features...")
feat = pd.read_excel("ipo_preipo_features.xlsx")          # pre-IPO snapshot + returns
pp = pd.read_excel("ipo_preprocessed.xlsx")               # has Range_Width_pct, Days_to_List, PostIPO_price_std

feat = feat.merge(
    pp[["COMPANY NAME", "Range_Width_pct", "Days_to_List", "PostIPO_price_std"]],
    on="COMPANY NAME", how="left",
)

# keep companies with a usable pre-IPO snapshot (revenue present)
feat = feat[feat["PreIPO_Revenue"].notna()].copy()

# impute remaining gaps with median (e.g. growth where prior FY missing, DE/coverage)
for col in CLUSTER_FEATURES:
    n_missing = feat[col].isna().sum()
    if n_missing:
        feat[col] = feat[col].fillna(feat[col].median())
        print(f"  imputed {n_missing} missing in {col}")

# winsorize the heavy-tailed columns (same spirit as original)
for col in ["PreIPO_ROE_pct", "PreIPO_ROCE_pct", "PreIPO_DE_pct",
            "PreIPO_Interest_Coverage", "PreIPO_Revenue_Growth", "PreIPO_PAT_Growth",
            "Range_Width_pct", "Days_to_List"]:
    feat[col] = winsorize_clip(feat[col])

print(f"  Cohort size: {len(feat)} companies")

X = RobustScaler().fit_transform(feat[CLUSTER_FEATURES])

# ── k selection ──────────────────────────────────────────────────────────────
print("\nSelecting k (2..10)...")
ks = range(2, 11)
inertia, sil, ch, db = [], [], [], []
for k in ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(X)
    inertia.append(km.inertia_)
    sil.append(silhouette_score(X, km.labels_))
    ch.append(calinski_harabasz_score(X, km.labels_))
    db.append(davies_bouldin_score(X, km.labels_))
    print(f"  k={k}: silhouette={sil[-1]:.3f}, CH={ch[-1]:.0f}, DB={db[-1]:.3f}")

fig, ax = plt.subplots(2, 2, figsize=(13, 9))
ax[0,0].plot(list(ks), inertia, "o-"); ax[0,0].set_title("Elbow (inertia)")
ax[0,1].plot(list(ks), sil, "o-", color="green"); ax[0,1].set_title("Silhouette")
ax[1,0].plot(list(ks), ch, "o-", color="purple"); ax[1,0].set_title("Calinski-Harabasz")
ax[1,1].plot(list(ks), db, "o-", color="red"); ax[1,1].set_title("Davies-Bouldin")
for a in ax.flat: a.set_xlabel("k")
plt.tight_layout(); plt.savefig("reclustering_k_selection.png", dpi=120); plt.close()

best_k = list(ks)[int(np.argmax(sil))]
print(f"  Best k by silhouette: {best_k}")

# ── Fit & compare ────────────────────────────────────────────────────────────
km = KMeans(n_clusters=best_k, n_init=10, random_state=RANDOM_STATE).fit(X)
agg = AgglomerativeClustering(n_clusters=best_k, linkage="ward").fit(X)
print(f"  K-Means silhouette: {silhouette_score(X, km.labels_):.3f}")
print(f"  Agglomerative silhouette: {silhouette_score(X, agg.labels_):.3f}")
feat["Cluster"] = km.labels_

pca = PCA(n_components=2, random_state=RANDOM_STATE).fit(X)
pcs = pca.transform(X)
plt.figure(figsize=(9, 7))
sc = plt.scatter(pcs[:,0], pcs[:,1], c=km.labels_, cmap="tab10", s=40, alpha=0.8)
plt.colorbar(sc, label="Cluster")
plt.title(f"Pre-IPO re-clustering (k={best_k})\nPC1 {pca.explained_variance_ratio_[0]*100:.0f}% / PC2 {pca.explained_variance_ratio_[1]*100:.0f}%")
plt.xlabel("PC1"); plt.ylabel("PC2")
plt.tight_layout(); plt.savefig("reclustering_pca.png", dpi=120); plt.close()

# ── Profiles ─────────────────────────────────────────────────────────────────
print("\nCluster profiles (pre-IPO feature means):")
profile = feat.groupby("Cluster")[CLUSTER_FEATURES].mean().round(2)
profile["N"] = feat.groupby("Cluster").size()
profile = profile[["N"] + CLUSTER_FEATURES]
print(profile.T.to_string())

ind_mix = pd.crosstab(feat["Cluster"], feat["Assigned Industry"], normalize="index").round(3)*100
yr_mix = pd.crosstab(feat["Cluster"], feat["Listing_Year"])

with pd.ExcelWriter("recluster_profiles.xlsx") as w:
    profile.to_excel(w, sheet_name="Feature_Means")
    ind_mix.to_excel(w, sheet_name="Industry_Pct")
    yr_mix.to_excel(w, sheet_name="ListingYear_Counts")

# ── Performance link + Kruskal-Wallis ────────────────────────────────────────
print("\nReturns per cluster (median):")
perf = feat.groupby("Cluster")[RETURN_COLS].median().round(2)
perf["N"] = feat.groupby("Cluster").size()
print(perf.to_string())

print("\nKruskal-Wallis (returns differ across clusters?):")
kw = []
for col in RETURN_COLS:
    groups = [g[col].dropna().values for _, g in feat.groupby("Cluster")]
    groups = [g for g in groups if len(g) > 0]
    if len(groups) >= 2:
        stat, p = kruskal(*groups)
        sig = "YES" if p < 0.05 else "no"
        kw.append({"Metric": col, "H": round(stat,2), "p": round(p,4), "sig": sig})
        print(f"  {col:20s}: H={stat:6.2f}, p={p:.4f}  {sig}")
kw_df = pd.DataFrame(kw)

with pd.ExcelWriter("recluster_performance.xlsx") as w:
    perf.to_excel(w, sheet_name="Median_Returns")
    kw_df.to_excel(w, sheet_name="KruskalWallis", index=False)

feat.to_excel("ipo_reclustered_preipo.xlsx", index=False)
print("\nDone. Saved outputs.")
