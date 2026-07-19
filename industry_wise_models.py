"""
Industry-wise modeling + binary (positive vs negative) listing target.

Two questions:
  A) Does the return->driver relationship differ BY INDUSTRY?
  B) Binary target (list positive vs negative) instead of raw return %.

Because each industry has only 22-47 companies, we DON'T fit an 11-feature model
per industry (that would overfit). Per industry we report:
  - base rate (share listing positive)
  - within-industry Spearman of each key driver vs Return_30d
  - a COMPACT logistic model (4 defensible drivers) with stratified CV AUC
And we compare against a single POOLED logistic model (with industry dummies).

Target: Positive_30d = (Return_30d > 0).

Outputs:
  industry_wise_drivers.xlsx     - per-industry driver correlations + base rates
  industry_wise_models.xlsx      - per-industry vs pooled CV AUC
  industry_wise_auc.png
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

RANDOM_STATE = 42

# compact, defensible driver set (avoid overfitting on small per-industry n)
COMPACT = ["PreIPO_Revenue_Growth", "PreIPO_ROE_pct", "Log_PreIPO_Revenue", "Company_Age_at_IPO"]
# fuller driver list for the within-industry correlation scan
DRIVERS = ["PreIPO_Revenue_Growth", "PreIPO_PAT_Growth", "PreIPO_ROE_pct",
           "PreIPO_ROCE_pct", "PreIPO_Operating_Margin", "PreIPO_DE_pct",
           "Log_PreIPO_Revenue", "Company_Age_at_IPO"]
TARGET = "Return_30d"


def winsorize_clip(s, lo=0.05, hi=0.95):
    return s.clip(s.quantile(lo), s.quantile(hi))


print("Loading...")
feat = pd.read_excel("ipo_preipo_features.xlsx")
for col in DRIVERS:
    if feat[col].isna().any():
        feat[col] = feat[col].fillna(feat[col].median())
for col in ["PreIPO_ROE_pct", "PreIPO_ROCE_pct", "PreIPO_DE_pct",
            "PreIPO_Revenue_Growth", "PreIPO_PAT_Growth"]:
    feat[col] = winsorize_clip(feat[col])

feat = feat.dropna(subset=[TARGET]).copy()
feat["Positive"] = (feat[TARGET] > 0).astype(int)
print(f"  n={len(feat)}, overall positive rate={feat['Positive'].mean():.1%}")

# ── A) within-industry driver correlations + base rates ─────────────────────
print("\nPer-industry base rate + within-industry Spearman(driver, Return_30d):")
corr_rows = []
for ind, g in feat.groupby("Assigned Industry"):
    row = {"Industry": ind, "N": len(g), "Positive_rate": round(g["Positive"].mean(), 3),
           "Median_Return_30d": round(g[TARGET].median(), 1)}
    for d in DRIVERS:
        sub = g[[d, TARGET]].dropna()
        if len(sub) >= 8:
            rho, p = spearmanr(sub[d], sub[TARGET])
            row[f"{d}"] = round(rho, 2)
            row[f"{d}_p"] = round(p, 3)
    corr_rows.append(row)
corr_df = pd.DataFrame(corr_rows).sort_values("N", ascending=False)
# compact view for console
show_cols = ["Industry", "N", "Positive_rate", "Median_Return_30d",
             "PreIPO_Revenue_Growth", "PreIPO_ROE_pct", "Log_PreIPO_Revenue"]
print(corr_df[show_cols].to_string(index=False))
corr_df.to_excel("industry_wise_drivers.xlsx", index=False)

# ── B) per-industry compact logistic vs pooled model (CV AUC) ───────────────
def cv_auc(X, y, n_splits):
    # guard: need both classes and enough per fold
    if y.nunique() < 2:
        return np.nan
    n_splits = max(2, min(n_splits, y.value_counts().min()))
    if n_splits < 2:
        return np.nan
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    pipe = make_pipeline(RobustScaler(), LogisticRegression(max_iter=1000, C=0.5))
    try:
        s = cross_val_score(pipe, X, y, cv=skf, scoring="roc_auc")
        return s.mean()
    except ValueError:
        return np.nan


print("\nPer-industry compact logistic model (4 drivers), stratified CV AUC:")
auc_rows = []
for ind, g in feat.groupby("Assigned Industry"):
    X, y = g[COMPACT].values, g["Positive"]
    auc = cv_auc(X, y, n_splits=5)
    auc_rows.append({"Model": f"Industry: {ind}", "N": len(g),
                     "Positive_rate": round(y.mean(), 3),
                     "CV_AUC": round(auc, 3) if pd.notna(auc) else None})
    print(f"  {ind:44s} n={len(g):3d}  AUC={auc:.3f}" if pd.notna(auc)
          else f"  {ind:44s} n={len(g):3d}  AUC=NA")

# pooled model: same 4 drivers, all industries, WITH industry dummies
ind_dum = pd.get_dummies(feat["Assigned Industry"], prefix="Ind", drop_first=True)
X_pool_nofe = feat[COMPACT].values
X_pool_fe = pd.concat([feat[COMPACT], ind_dum], axis=1).values
y_pool = feat["Positive"]
auc_pool_nofe = cv_auc(X_pool_nofe, y_pool, 5)
auc_pool_fe = cv_auc(X_pool_fe, y_pool, 5)
print(f"\n  POOLED (4 drivers, no industry)      n={len(feat)}  AUC={auc_pool_nofe:.3f}")
print(f"  POOLED (4 drivers + industry dummies) n={len(feat)}  AUC={auc_pool_fe:.3f}")
print("  (AUC 0.5 = coin flip, 1.0 = perfect)")

auc_rows.append({"Model": "POOLED (no industry)", "N": len(feat),
                 "Positive_rate": round(y_pool.mean(), 3), "CV_AUC": round(auc_pool_nofe, 3)})
auc_rows.append({"Model": "POOLED (+ industry dummies)", "N": len(feat),
                 "Positive_rate": round(y_pool.mean(), 3), "CV_AUC": round(auc_pool_fe, 3)})
auc_df = pd.DataFrame(auc_rows)
auc_df.to_excel("industry_wise_models.xlsx", index=False)

# ── plot ─────────────────────────────────────────────────────────────────────
plot_df = auc_df.dropna(subset=["CV_AUC"])
plt.figure(figsize=(11, 6))
colors = ["steelblue" if not m.startswith("POOLED") else "darkorange" for m in plot_df["Model"]]
plt.barh(plot_df["Model"], plot_df["CV_AUC"], color=colors)
plt.axvline(0.5, color="red", ls="--", label="coin flip (0.5)")
plt.xlabel("5-fold stratified CV AUC")
plt.title("Predicting a POSITIVE 30-day listing — per-industry vs pooled\n(blue = single industry, orange = pooled)")
plt.legend(); plt.tight_layout(); plt.savefig("industry_wise_auc.png", dpi=120); plt.close()

print("\nSaved: industry_wise_drivers.xlsx, industry_wise_models.xlsx, industry_wise_auc.png")
