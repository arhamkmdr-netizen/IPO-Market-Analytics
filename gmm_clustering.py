"""
Gaussian Mixture Model (GMM) clustering on the PRE-IPO snapshot cohort.
(Roadmap item 3.2)

Unlike K-Means (hard, spherical clusters), GMM gives:
  - soft / probabilistic membership (P(company in each cluster))
  - elliptical clusters (each component has its own covariance shape)
  - principled model selection via BIC/AIC

Same 230-company pre-IPO cohort + feature set as reclustering_preipo.py,
so results are directly comparable to the K-Means re-clustering.

Outputs:
  gmm_model_selection.png       - BIC/AIC vs n_components x covariance type
  gmm_pca.png                   - PCA scatter colored by GMM component + uncertainty
  ipo_gmm_clustered.xlsx        - companies + hard label + soft probabilities
  gmm_profiles.xlsx             - feature means, industry mix, listing-year mix
  gmm_performance.xlsx          - median returns per component + Kruskal-Wallis
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import kruskal
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler

RANDOM_STATE = 42

CLUSTER_FEATURES = [
    "Company_Age_at_IPO", "Log_Issue_Price", "Range_Width_pct", "Days_to_List",
    "PreIPO_Revenue_Growth", "PreIPO_PAT_Growth", "PreIPO_Operating_Margin",
    "PreIPO_ROE_pct", "PreIPO_ROCE_pct", "PreIPO_DE_pct",
    "PreIPO_Interest_Coverage", "PreIPO_Assets_Turnover", "Log_PreIPO_Revenue",
]
RETURN_COLS = ["Return_1d", "Return_30d", "Return_60d"]


def winsorize_clip(s, lo=0.05, hi=0.95):
    return s.clip(s.quantile(lo), s.quantile(hi))


# ── Assemble the same feature table as the K-Means re-clustering ─────────────
print("Assembling pre-IPO clustering features...")
feat = pd.read_excel("ipo_preipo_features.xlsx")
pp = pd.read_excel("ipo_preprocessed.xlsx")

feat = feat.merge(
    pp[["COMPANY NAME", "Range_Width_pct", "Days_to_List", "PostIPO_price_std"]],
    on="COMPANY NAME", how="left",
)
feat = feat[feat["PreIPO_Revenue"].notna()].copy()

for col in CLUSTER_FEATURES:
    n_missing = feat[col].isna().sum()
    if n_missing:
        feat[col] = feat[col].fillna(feat[col].median())

for col in ["PreIPO_ROE_pct", "PreIPO_ROCE_pct", "PreIPO_DE_pct",
            "PreIPO_Interest_Coverage", "PreIPO_Revenue_Growth", "PreIPO_PAT_Growth",
            "Range_Width_pct", "Days_to_List"]:
    feat[col] = winsorize_clip(feat[col])

print(f"  Cohort size: {len(feat)} companies, {len(CLUSTER_FEATURES)} features")

X = RobustScaler().fit_transform(feat[CLUSTER_FEATURES])

# ── Model selection: BIC / AIC over n_components and covariance type ─────────
print("\nSelecting GMM (n_components 1..10 x covariance type)...")
cov_types = ["full", "tied", "diag", "spherical"]
n_range = range(1, 11)
records = []
best = {"bic": np.inf}
for cov in cov_types:
    for n in n_range:
        gm = GaussianMixture(n_components=n, covariance_type=cov,
                             n_init=5, random_state=RANDOM_STATE, max_iter=300)
        gm.fit(X)
        bic, aic = gm.bic(X), gm.aic(X)
        records.append({"cov": cov, "n": n, "BIC": bic, "AIC": aic,
                        "converged": gm.converged_})
        if bic < best["bic"] and n >= 2:
            best = {"bic": bic, "cov": cov, "n": n, "model": gm}
sel = pd.DataFrame(records)

print(f"  Best by BIC (n>=2): covariance='{best['cov']}', n_components={best['n']}, BIC={best['bic']:.0f}")

# plot BIC/AIC curves
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
for cov in cov_types:
    s = sel[sel["cov"] == cov]
    ax[0].plot(s["n"], s["BIC"], "o-", label=cov)
    ax[1].plot(s["n"], s["AIC"], "o-", label=cov)
ax[0].set_title("BIC (lower = better)"); ax[0].set_xlabel("n_components"); ax[0].legend()
ax[1].set_title("AIC (lower = better)"); ax[1].set_xlabel("n_components"); ax[1].legend()
ax[0].axvline(best["n"], color="grey", ls="--", alpha=0.6)
plt.tight_layout(); plt.savefig("gmm_model_selection.png", dpi=120); plt.close()

# ── Fit the selected model, get hard labels + soft probabilities ────────────
gm = best["model"]
labels = gm.predict(X)
proba = gm.predict_proba(X)
feat["GMM_Cluster"] = labels
for k in range(best["n"]):
    feat[f"P_cluster_{k}"] = proba[:, k].round(3)
feat["Max_Membership_Prob"] = proba.max(axis=1).round(3)

# how confident are the assignments? (soft clustering diagnostic)
conf = feat["Max_Membership_Prob"]
n_uncertain = int((conf < 0.8).sum())
print(f"\nSoft-membership confidence: median={conf.median():.2f}, "
      f"{n_uncertain}/{len(feat)} companies below 0.8 (ambiguous)")

sil = silhouette_score(X, labels) if best["n"] >= 2 else float("nan")
print(f"Silhouette of GMM hard labels: {sil:.3f}")

# ── PCA visualization (point size = assignment confidence) ──────────────────
pca = PCA(n_components=2, random_state=RANDOM_STATE).fit(X)
pcs = pca.transform(X)
plt.figure(figsize=(9, 7))
sizes = 20 + 80 * (conf.values - conf.min()) / (conf.max() - conf.min() + 1e-9)
sc = plt.scatter(pcs[:, 0], pcs[:, 1], c=labels, cmap="tab10", s=sizes, alpha=0.75,
                 edgecolors="k", linewidths=0.3)
plt.colorbar(sc, label="GMM component")
plt.title(f"GMM soft clustering (n={best['n']}, cov={best['cov']})\n"
          f"PC1 {pca.explained_variance_ratio_[0]*100:.0f}% / "
          f"PC2 {pca.explained_variance_ratio_[1]*100:.0f}% — point size = membership confidence")
plt.xlabel("PC1"); plt.ylabel("PC2")
plt.tight_layout(); plt.savefig("gmm_pca.png", dpi=120); plt.close()

# ── Profiles ─────────────────────────────────────────────────────────────────
print("\nCluster profiles (pre-IPO feature means):")
profile = feat.groupby("GMM_Cluster")[CLUSTER_FEATURES].mean().round(2)
profile["N"] = feat.groupby("GMM_Cluster").size()
profile = profile[["N"] + CLUSTER_FEATURES]
print(profile.T.to_string())

ind_mix = pd.crosstab(feat["GMM_Cluster"], feat["Assigned Industry"], normalize="index").round(3) * 100
yr_mix = pd.crosstab(feat["GMM_Cluster"], feat["Listing_Year"])

with pd.ExcelWriter("gmm_profiles.xlsx") as w:
    profile.to_excel(w, sheet_name="Feature_Means")
    ind_mix.to_excel(w, sheet_name="Industry_Pct")
    yr_mix.to_excel(w, sheet_name="ListingYear_Counts")

# ── Performance link + Kruskal-Wallis ────────────────────────────────────────
perf_cols = RETURN_COLS + ["PostIPO_price_std"]
print("\nReturns per cluster (median):")
perf = feat.groupby("GMM_Cluster")[perf_cols].median().round(2)
perf["N"] = feat.groupby("GMM_Cluster").size()
print(perf.to_string())

print("\nKruskal-Wallis (returns differ across GMM clusters?):")
kw = []
for col in perf_cols:
    groups = [g[col].dropna().values for _, g in feat.groupby("GMM_Cluster")]
    groups = [g for g in groups if len(g) > 0]
    if len(groups) >= 2:
        stat, p = kruskal(*groups)
        sig = "YES" if p < 0.05 else "no"
        kw.append({"Metric": col, "H": round(stat, 2), "p": round(p, 4), "sig": sig})
        print(f"  {col:20s}: H={stat:6.2f}, p={p:.4f}  {sig}")
kw_df = pd.DataFrame(kw)

with pd.ExcelWriter("gmm_performance.xlsx") as w:
    perf.to_excel(w, sheet_name="Median_Returns")
    kw_df.to_excel(w, sheet_name="KruskalWallis", index=False)
    sel.to_excel(w, sheet_name="Model_Selection", index=False)

feat.to_excel("ipo_gmm_clustered.xlsx", index=False)
print("\nDone. Saved: gmm_model_selection.png, gmm_pca.png, ipo_gmm_clustered.xlsx, "
      "gmm_profiles.xlsx, gmm_performance.xlsx")
