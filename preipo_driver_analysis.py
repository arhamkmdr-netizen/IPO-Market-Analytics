"""
Pre-IPO snapshot feature set + driver analysis.

Instead of averaging 2022-2026 financials (which mixes in POST-listing years),
this uses each company's financials from the last fiscal year that ended BEFORE
its listing date (the RHP "pre-IPO" snapshot). It then tests which pre-IPO
fundamentals are associated with listing returns.

Cohort: only companies whose pre-IPO FY is present in our 2022-2026 data
(231 companies; 199 also have the prior FY for growth metrics).

Outputs:
  ipo_preipo_features.xlsx        - pre-IPO snapshot feature table
  preipo_feature_correlation.xlsx - Spearman corr of features vs returns
  preipo_feature_importance.xlsx  - Random Forest + permutation importance
  preipo_correlation_heatmap.png
  preipo_feature_importance.png
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_score

RANDOM_STATE = 42

FINANCIAL_COLS = [
    "Total Operating Revenue", "Operating Profit", "Profit before Income Tax",
    "Net Profit/Loss for the Period", "Return on Equity (ROE) (%)",
    "Return on Capital Employed (%)", "Debt / Equity (%)",
    "Interest Coverage Ratio (x)", "Assets Turnover (x)",
]

# Same confirmed bad early-year figures as preprocessing
BAD_YEARS = {
    "Indiqube Spaces Limited":                  [2022],
    "FiveStar Business Finance Limited":         [2022, 2023],
    "BLS E- Services Limited":                   [2022],
    "Jain Resource Recycling Limited":           [2022],
    "Sri Lotus Developers and Realty Limited":   [2022],
}

RETURN_COLS = ["Return_1d", "Return_7d", "Return_15d", "Return_21d",
               "Return_30d", "Return_45d", "Return_60d"]


def winsorize_clip(s, lo=0.05, hi=0.95):
    ql, qh = s.quantile(lo), s.quantile(hi)
    return s.clip(ql, qh)


# ── Load & prep ──────────────────────────────────────────────────────────────
print("Loading data...")
merged = pd.read_excel("ipo_merged_dataset.xlsx")
merged = merged[merged["Matched Financial Company"].notna()].copy()

# nullify bad early-year financials
for company, years in BAD_YEARS.items():
    mask = (merged["COMPANY NAME"] == company) & (merged["Year"].isin(years))
    merged.loc[mask, FINANCIAL_COLS] = np.nan

dates = pd.read_csv("ipo_listing_dates.csv")
dates["Listing Date"] = pd.to_datetime(dates["Listing Date"])
dates["Listing_Year"] = dates["Listing Date"].dt.year
dates["Listing_Month"] = dates["Listing Date"].dt.month
dates["PreIPO_FY"] = dates.apply(
    lambda r: r["Listing_Year"] if r["Listing_Month"] >= 4 else r["Listing_Year"] - 1, axis=1
)

# ── Build pre-IPO snapshot per company ──────────────────────────────────────
print("Building pre-IPO snapshot features...")
rows = []
fin_by_company = {c: g for c, g in merged.groupby("COMPANY NAME")}

for _, d in dates.iterrows():
    company = d["COMPANY NAME"]
    fy = d["PreIPO_FY"]
    g = fin_by_company.get(company)
    if g is None:
        continue
    snap = g[g["Year"] == fy]
    if snap.empty or snap["Total Operating Revenue"].isna().all():
        continue  # pre-IPO FY not available in our data
    snap = snap.iloc[0]
    prior = g[g["Year"] == fy - 1]
    prior = prior.iloc[0] if (not prior.empty and prior["Total Operating Revenue"].notna().any()) else None

    rev = snap["Total Operating Revenue"]
    op = snap["Operating Profit"]
    pat = snap["Net Profit/Loss for the Period"]

    # pre-IPO growth (1-yr) where prior available
    rev_growth = pat_growth = np.nan
    if prior is not None:
        pr_rev, pr_pat = prior["Total Operating Revenue"], prior["Net Profit/Loss for the Period"]
        if pd.notna(pr_rev) and pr_rev > 0 and pd.notna(rev):
            rev_growth = rev / pr_rev - 1
        if pd.notna(pr_pat) and abs(pr_pat) > 1e-9 and pd.notna(pat):
            pat_growth = (pat - pr_pat) / abs(pr_pat)

    # company age at IPO
    inc = g["Incorporation Date"].dropna()
    inc_year = np.nan
    if not inc.empty:
        try:
            inc_year = int(str(inc.iloc[0])[:4])
        except (ValueError, TypeError):
            pass
    age = d["Listing_Year"] - inc_year if pd.notna(inc_year) else np.nan

    rows.append({
        "COMPANY NAME": company,
        "Symbol": d["Symbol"],
        "Listing_Year": d["Listing_Year"],
        "PreIPO_FY": fy,
        "ISSUE PRICE": None,  # filled later from preprocessed
        "PreIPO_Revenue": rev,
        "PreIPO_Net_Profit": pat,
        "PreIPO_Operating_Margin": (op / rev) if (pd.notna(op) and pd.notna(rev) and rev > 0) else np.nan,
        "PreIPO_ROE_pct": snap["Return on Equity (ROE) (%)"],
        "PreIPO_ROCE_pct": snap["Return on Capital Employed (%)"],
        "PreIPO_DE_pct": snap["Debt / Equity (%)"],
        "PreIPO_Interest_Coverage": snap["Interest Coverage Ratio (x)"],
        "PreIPO_Assets_Turnover": snap["Assets Turnover (x)"],
        "PreIPO_Revenue_Growth": rev_growth,
        "PreIPO_PAT_Growth": pat_growth,
        "Company_Age_at_IPO": age,
    })

feat = pd.DataFrame(rows)
print(f"  Pre-IPO snapshot built for {len(feat)} companies")

# bring in issue price + returns + industry from preprocessed/targets
pp = pd.read_excel("ipo_preprocessed.xlsx")
feat = feat.merge(pp[["COMPANY NAME", "ISSUE PRICE", "Assigned Industry"]], on="COMPANY NAME",
                  how="left", suffixes=("_drop", ""))
feat.drop(columns=[c for c in feat.columns if c.endswith("_drop")], inplace=True)
feat["Log_Issue_Price"] = np.log1p(feat["ISSUE PRICE"])
feat["Log_PreIPO_Revenue"] = np.log1p(feat["PreIPO_Revenue"].clip(lower=0))

targets = pd.read_csv("ipo_targets.csv")
feat = feat.merge(targets[["COMPANY NAME"] + RETURN_COLS], on="COMPANY NAME", how="left")

feat.to_excel("ipo_preipo_features.xlsx", index=False)
print(f"  Saved ipo_preipo_features.xlsx ({feat.shape[0]} rows x {feat.shape[1]} cols)")

# ── Step 7: Spearman correlation vs returns ─────────────────────────────────
print("\nSpearman correlation: pre-IPO features vs returns")
FEATURES = [
    "PreIPO_ROE_pct", "PreIPO_ROCE_pct", "PreIPO_Operating_Margin",
    "PreIPO_Revenue_Growth", "PreIPO_PAT_Growth", "PreIPO_DE_pct",
    "PreIPO_Interest_Coverage", "PreIPO_Assets_Turnover",
    "Log_PreIPO_Revenue", "Log_Issue_Price", "Company_Age_at_IPO",
]
ret_test = ["Return_1d", "Return_30d", "Return_60d"]

corr_rows = []
for f in FEATURES:
    row = {"Feature": f}
    for r in ret_test:
        sub = feat[[f, r]].dropna()
        if len(sub) > 10:
            rho, p = spearmanr(sub[f], sub[r])
            row[f"{r}_rho"] = round(rho, 3)
            row[f"{r}_p"] = round(p, 4)
            row[f"{r}_n"] = len(sub)
    corr_rows.append(row)
corr_df = pd.DataFrame(corr_rows)
print(corr_df.to_string(index=False))
corr_df.to_excel("preipo_feature_correlation.xlsx", index=False)

# heatmap of rho
plt.figure(figsize=(7, 7))
hm = corr_df.set_index("Feature")[[f"{r}_rho" for r in ret_test]]
hm.columns = ret_test
plt.imshow(hm.values, cmap="RdBu_r", vmin=-0.3, vmax=0.3, aspect="auto")
plt.colorbar(label="Spearman rho")
plt.xticks(range(len(ret_test)), ret_test)
plt.yticks(range(len(hm.index)), hm.index)
for i in range(len(hm.index)):
    for j in range(len(ret_test)):
        plt.text(j, i, f"{hm.values[i,j]:.2f}", ha="center", va="center", fontsize=8)
plt.title("Pre-IPO fundamentals vs listing returns (Spearman)")
plt.tight_layout(); plt.savefig("preipo_correlation_heatmap.png", dpi=120); plt.close()

# ── Step 8: Random Forest feature importance (target Return_30d) ─────────────
print("\nRandom Forest feature importance (target = Return_30d)")
ind_dummies = pd.get_dummies(feat["Assigned Industry"], prefix="Ind", drop_first=True)
X_full = pd.concat([feat[FEATURES], ind_dummies], axis=1)
y = feat["Return_30d"]

model_df = pd.concat([X_full, y], axis=1).dropna()
X = model_df[X_full.columns]
yv = model_df["Return_30d"]
print(f"  Model sample: {len(model_df)} companies, {X.shape[1]} features")

rf = RandomForestRegressor(n_estimators=400, random_state=RANDOM_STATE, max_depth=6, min_samples_leaf=5)
rf.fit(X, yv)
r2_cv = cross_val_score(rf, X, yv, cv=5, scoring="r2")
print(f"  5-fold CV R^2: mean={r2_cv.mean():.3f} (per-fold: {np.round(r2_cv,2)})")

perm = permutation_importance(rf, X, yv, n_repeats=30, random_state=RANDOM_STATE)
imp = pd.DataFrame({
    "Feature": X.columns,
    "RF_Importance": rf.feature_importances_,
    "Permutation_Importance": perm.importances_mean,
}).sort_values("Permutation_Importance", ascending=False)
print(imp.to_string(index=False))
imp.to_excel("preipo_feature_importance.xlsx", index=False)

top = imp.head(12).iloc[::-1]
plt.figure(figsize=(9, 6))
plt.barh(top["Feature"], top["Permutation_Importance"], color="teal")
plt.title("Pre-IPO driver importance for Return_30d (permutation)")
plt.xlabel("Permutation importance")
plt.tight_layout(); plt.savefig("preipo_feature_importance.png", dpi=120); plt.close()

print("\nDone. Outputs saved.")
