"""
Hierarchical / incremental regression — how much does each feature BLOCK add?
(Roadmap items 1.1, 1.2, 6.1)

Nested models, each adding one block of predictors:
  M1  Fundamentals
  M2  + Issue structure
  M3  + Industry (sector fixed effects)
  M4  + Market timing (listing-year fixed effects)

For each model we report:
  - In-sample R2  and  incremental Delta-R2 vs the previous model
  - F-change test: is the block's Delta-R2 statistically significant?  (p_change)
  - 5-fold CV R2: does the added block actually improve out-of-sample fit?

In-sample R2 can only go up as features are added; the F-change test says whether
that rise is more than chance, and the CV R2 says whether it generalizes.

Target = Return_1d / Return_30d / Return_60d.

Outputs:
  hierarchical_regression.xlsx   - per-horizon block-by-block table
  hierarchical_regression.png    - CV R2 vs in-sample R2 across M1..M4 (Return_30d)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import f as f_dist
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

RANDOM_STATE = 42

FUNDAMENTALS = [
    "PreIPO_ROE_pct", "PreIPO_ROCE_pct", "PreIPO_Operating_Margin",
    "PreIPO_Revenue_Growth", "PreIPO_PAT_Growth", "PreIPO_DE_pct",
    "PreIPO_Interest_Coverage", "PreIPO_Assets_Turnover", "Log_PreIPO_Revenue",
    "Company_Age_at_IPO",
]
ISSUE_STRUCTURE = ["Log_Issue_Price", "Range_Width_pct", "Days_to_List"]
HORIZONS = ["Return_1d", "Return_30d", "Return_60d"]


def winsorize_clip(s, lo=0.05, hi=0.95):
    return s.clip(s.quantile(lo), s.quantile(hi))


# ── Assemble feature table ───────────────────────────────────────────────────
print("Loading features...")
feat = pd.read_excel("ipo_preipo_features.xlsx")
pp = pd.read_excel("ipo_preprocessed.xlsx")
feat = feat.merge(pp[["COMPANY NAME", "Range_Width_pct", "Days_to_List"]],
                  on="COMPANY NAME", how="left")

# impute + winsorize (same conventions as other scripts)
num_cols = FUNDAMENTALS + ISSUE_STRUCTURE
for col in num_cols:
    if feat[col].isna().any():
        feat[col] = feat[col].fillna(feat[col].median())
for col in ["PreIPO_ROE_pct", "PreIPO_ROCE_pct", "PreIPO_DE_pct",
            "PreIPO_Interest_Coverage", "PreIPO_Revenue_Growth", "PreIPO_PAT_Growth",
            "Range_Width_pct", "Days_to_List"]:
    feat[col] = winsorize_clip(feat[col])

industry_dummies = pd.get_dummies(feat["Assigned Industry"], prefix="Ind", drop_first=True)
year_dummies = pd.get_dummies(feat["Listing_Year"], prefix="Yr", drop_first=True)

# Incremental feature blocks
BLOCKS = [
    ("M1  Fundamentals",              FUNDAMENTALS),
    ("M2  + Issue structure",         ISSUE_STRUCTURE),
    ("M3  + Industry (sector FE)",    list(industry_dummies.columns)),
    ("M4  + Market timing (year FE)", list(year_dummies.columns)),
]

full_design = pd.concat([feat[num_cols], industry_dummies, year_dummies], axis=1).astype(float)

cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
all_rows = []

for horizon in HORIZONS:
    print(f"\n=== {horizon} ===")
    df = pd.concat([full_design, feat[horizon]], axis=1).dropna()
    y = df[horizon].values
    n = len(df)

    cols_so_far = []
    prev_r2_in = 0.0
    prev_k = 0
    for label, block in BLOCKS:
        cols_so_far = cols_so_far + block
        X = df[cols_so_far].values
        k = X.shape[1]

        # in-sample R2 (plain OLS)
        lr = LinearRegression().fit(X, y)
        r2_in = lr.score(X, y)

        # F-change test for the incremental block
        df_num = k - prev_k                 # params added
        df_den = n - k - 1                  # residual df of the larger model
        if df_num > 0 and df_den > 0 and (1 - r2_in) > 1e-12:
            f_change = ((r2_in - prev_r2_in) / df_num) / ((1 - r2_in) / df_den)
            p_change = f_dist.sf(f_change, df_num, df_den)
        else:
            f_change = p_change = np.nan

        # 5-fold CV R2 (scaled pipeline, no leakage)
        pipe = make_pipeline(RobustScaler(), LinearRegression())
        cv_r2 = cross_val_score(pipe, X, y, cv=cv, scoring="r2")

        all_rows.append({
            "Horizon": horizon, "Model": label, "N": n, "n_features": k,
            "R2_insample": round(r2_in, 4),
            "Delta_R2": round(r2_in - prev_r2_in, 4),
            "F_change": round(f_change, 3) if pd.notna(f_change) else None,
            "p_change": round(p_change, 4) if pd.notna(p_change) else None,
            "sig": ("YES" if (pd.notna(p_change) and p_change < 0.05) else "no"),
            "CV_R2_mean": round(cv_r2.mean(), 4),
            "CV_R2_std": round(cv_r2.std(), 4),
        })
        print(f"  {label:32s} R2={r2_in:.3f}  dR2={r2_in-prev_r2_in:+.3f}  "
              f"p_change={p_change if pd.isna(p_change) else round(p_change,3)}  "
              f"CV_R2={cv_r2.mean():+.3f}")
        prev_r2_in, prev_k = r2_in, k

res = pd.DataFrame(all_rows)
res.to_excel("hierarchical_regression.xlsx", index=False)

# ── Plot (Return_30d): in-sample vs CV R2 across the ladder ──────────────────
sub = res[res["Horizon"] == "Return_30d"]
x = np.arange(len(sub))
plt.figure(figsize=(10, 6))
plt.plot(x, sub["R2_insample"], "o-", label="In-sample R²", color="steelblue")
plt.plot(x, sub["CV_R2_mean"], "s--", label="5-fold CV R²", color="darkorange")
plt.axhline(0, color="black", lw=0.8)
plt.xticks(x, sub["Model"], rotation=20, ha="right")
plt.ylabel("R²")
plt.title("Hierarchical regression — Return_30d\nin-sample rises with every block; CV R² shows what actually generalizes")
plt.legend(); plt.tight_layout(); plt.savefig("hierarchical_regression.png", dpi=120); plt.close()

print("\nSaved: hierarchical_regression.xlsx, hierarchical_regression.png")
