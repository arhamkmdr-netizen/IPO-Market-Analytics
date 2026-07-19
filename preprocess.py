"""
IPO Research — Preprocessing Pipeline
Implements the approved plan (Steps 1-11):
  1. Filter to 302 matched companies
  2. Data quality fixes
  3. Derive financial features from 5-year long format
  4. Derive Company Age
  5. Collapse to 1 row per company
  6. Fill IPO placeholder columns
  7. Winsorize outliers
  8. Impute missing values
  9. One-hot encode Assigned Industry
  10. Separate X (features) from Y (targets)
  11. Scale with RobustScaler

Outputs:
  ipo_preprocessed.xlsx     — 302 rows, all derived columns, pre-scaling
  ipo_features_scaled.csv   — 302 rows x ~22 scaled features (X)
  ipo_targets.csv           — 302 rows x return/outcome columns (Y)
"""

import numpy as np
import pandas as pd
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import RobustScaler

# ── helpers ──────────────────────────────────────────────────────────────────

FINANCIAL_COLS = [
    "Total Operating Revenue",
    "Operating Profit",
    "Profit before Income Tax",
    "Net Profit/Loss for the Period",
    "Return on Equity (ROE) (%)",
    "Return on Capital Employed (%)",
    "Debt / Equity (%)",
    "Interest Coverage Ratio (x)",
    "Assets Turnover (x)",
]

IPO_LEVEL_COLS = [
    "COMPANY NAME", "Symbol", "Assigned Industry",
    "ISSUE PRICE", "Range_Width_pct", "Days_to_List", "PostIPO_price_std",
    "Return_1d", "Return_7d", "Return_15d",
    "Return_21d", "Return_30d", "Return_45d", "Return_60d",
]

TARGET_COLS = [
    "Return_1d", "Return_7d", "Return_15d",
    "Return_21d", "Return_30d", "Return_45d", "Return_60d",
    "PostIPO_price_std",
]


def revenue_cagr(series, years):
    """CAGR of revenue using first and last valid values."""
    valid = [(y, v) for y, v in zip(years, series) if pd.notna(v) and v > 0]
    if len(valid) < 2:
        return np.nan
    y0, v0 = valid[0]
    yn, vn = valid[-1]
    n = yn - y0
    if n <= 0:
        return np.nan
    return (vn / v0) ** (1 / n) - 1


def pat_cagr(series, years):
    """
    PAT growth rate. Geometric CAGR only when BOTH endpoints are positive
    (otherwise the formula produces complex/undefined values, e.g. profit->loss).
    Falls back to an absolute growth proxy in all other cases.
    """
    valid = [(y, v) for y, v in zip(years, series) if pd.notna(v)]
    if len(valid) < 2:
        return np.nan
    y0, v0 = valid[0]
    yn, vn = valid[-1]
    n = yn - y0
    if n <= 0:
        return np.nan
    if v0 > 0 and vn > 0:
        # both positive -> standard geometric CAGR
        return (vn / v0) ** (1 / n) - 1
    # any non-positive endpoint -> absolute growth proxy
    if abs(v0) < 1e-9:
        return np.nan
    return (vn - v0) / abs(v0)


def latest_valid(series, years):
    """Most recent non-null value."""
    pairs = [(y, v) for y, v in zip(years, series) if pd.notna(v)]
    return pairs[-1][1] if pairs else np.nan


def winsorize_col(series, limits=(0.05, 0.05)):
    """Winsorize a pandas Series; return Series."""
    arr = winsorize(series.dropna(), limits=limits)
    result = series.copy()
    result[series.notna()] = arr
    return result


# ── Step 1: load and filter to matched companies ──────────────────────────────

print("Step 1 — Loading and filtering to 302 matched companies...")
df = pd.read_excel("ipo_merged_dataset.xlsx")
df = df[df["Matched Financial Company"].notna()].copy()
print(f"  Rows: {len(df)}, Companies: {df['COMPANY NAME'].nunique()}")

# ── Step 2a: nullify confirmed bad early-year figures ────────────────────────

print("Step 2a — Nullifying bad early-year financial figures...")
# Each value is a list of years whose figures reflect a wrong/smaller pre-restructuring
# entity (verified via web search). FiveStar has TWO bad early years (2022 & 2023);
# its 2022->2023 step showed no jump because both were equally wrong.
BAD_YEARS = {
    "Indiqube Spaces Limited":                  [2022],
    "FiveStar Business Finance Limited":         [2022, 2023],
    "BLS E- Services Limited":                   [2022],
    "Jain Resource Recycling Limited":           [2022],
    "Sri Lotus Developers and Realty Limited":   [2022],
}
for company, bad_years in BAD_YEARS.items():
    mask = (df["COMPANY NAME"] == company) & (df["Year"].isin(bad_years))
    df.loc[mask, FINANCIAL_COLS] = np.nan
    print(f"  Nullified {company} year(s) {bad_years}: {mask.sum()} row(s)")

# ── Step 2b: fix Days_to_List for Ruchi Soya ────────────────────────────────

print("Step 2b — Fixing Days_to_List for Ruchi Soya Industries Limited...")
df.loc[df["COMPANY NAME"] == "Ruchi Soya Industries Limited", "Days_to_List"] = np.nan

# ── Step 2c: fix UNPARSED industry ──────────────────────────────────────────

print("Step 2c — Fixing UNPARSED industry...")
df.loc[df["Assigned Industry"] == "UNPARSED", "Assigned Industry"] = \
    "Infrastructure, Construction & Utilities"

# ── Step 3: derive per-company financial features ────────────────────────────

print("Step 3 — Deriving financial features from multi-year data...")

derived_rows = []
for company, grp in df.groupby("COMPANY NAME"):
    grp = grp.sort_values("Year")
    years  = grp["Year"].tolist()
    rev    = grp["Total Operating Revenue"].tolist()
    pat    = grp["Net Profit/Loss for the Period"].tolist()
    op     = grp["Operating Profit"].tolist()

    # operating margin per year (only where both are valid and revenue > 0)
    margins = [
        o / r for o, r in zip(op, rev)
        if pd.notna(o) and pd.notna(r) and r > 0
    ]

    row = {
        "COMPANY NAME":          company,
        "Revenue_CAGR":          revenue_cagr(rev, years),
        "PAT_CAGR":              pat_cagr(pat, years),
        "Avg_Operating_Margin":  np.nanmean(margins) if margins else np.nan,
        "Avg_ROE_pct":           grp["Return on Equity (ROE) (%)"].mean(),
        "Avg_ROCE_pct":          grp["Return on Capital Employed (%)"].mean(),
        "Avg_DE_pct":            grp["Debt / Equity (%)"].mean(),
        "Avg_Interest_Coverage": grp["Interest Coverage Ratio (x)"].mean(),
        "Avg_Assets_Turnover":   grp["Assets Turnover (x)"].mean(),
        "Latest_Revenue":        latest_valid(rev, years),
        "Latest_Net_Profit":     latest_valid(pat, years),
        "Latest_ROE":            latest_valid(grp["Return on Equity (ROE) (%)"].tolist(), years),
        "Latest_DE":             latest_valid(grp["Debt / Equity (%)"].tolist(), years),
    }
    derived_rows.append(row)

derived_df = pd.DataFrame(derived_rows)
print(f"  Derived features for {len(derived_df)} companies")

# ── Step 4: derive Company Age ───────────────────────────────────────────────

print("Step 4 — Deriving Company Age...")

def parse_inc_year(val):
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    # formats: YYYY-MM-DD, YYYY-MM, YYYY
    try:
        return int(s[:4])
    except (ValueError, IndexError):
        return np.nan

inc_dates = (
    df[["COMPANY NAME", "Incorporation Date"]]
    .dropna(subset=["Incorporation Date"])
    .drop_duplicates("COMPANY NAME")
    .set_index("COMPANY NAME")["Incorporation Date"]
)
inc_year_map = {c: parse_inc_year(v) for c, v in inc_dates.items()}
derived_df["Inc_Year"] = derived_df["COMPANY NAME"].map(inc_year_map)
derived_df["Company_Age"] = 2024 - derived_df["Inc_Year"]
derived_df.drop(columns="Inc_Year", inplace=True)

# ── Step 5: collapse IPO-level columns to 1 row per company ─────────────────

print("Step 5 — Collapsing to 1 row per company...")

ipo_level = (
    df[IPO_LEVEL_COLS]
    .groupby("COMPANY NAME")
    .first()
    .reset_index()
)

collapsed = ipo_level.merge(derived_df, on="COMPANY NAME", how="left")
print(f"  Collapsed shape: {collapsed.shape}  (expected 302 rows)")

# ── Step 6: fill IPO placeholder columns ────────────────────────────────────

print("Step 6 — Filling IPO placeholder columns...")
collapsed["Revenue"]                          = collapsed["Latest_Revenue"]
collapsed["net income"]                       = collapsed["Latest_Net_Profit"]
collapsed["Return on Equity (ROE)"]           = collapsed["Latest_ROE"]
collapsed["Debt-to-Equity Ratio"]             = collapsed["Latest_DE"]
collapsed["leverage (D/E) ratio"]             = collapsed["Latest_DE"]
collapsed["Revenue Growth Rate"]              = collapsed["Revenue_CAGR"]
collapsed["Profit Growth Rate (PAT Growth)"]  = collapsed["PAT_CAGR"]
collapsed["Company Age"]                      = collapsed["Company_Age"]

# ── Step 7: winsorize outliers ───────────────────────────────────────────────

print("Step 7 — Winsorizing outliers...")
WINSOR_COLS = [
    "Avg_ROE_pct", "Avg_ROCE_pct", "Avg_DE_pct",
    "Avg_Interest_Coverage", "Revenue_CAGR", "PAT_CAGR",
]
for col in WINSOR_COLS:
    before = collapsed[col].describe()[["min", "max"]].to_dict()
    collapsed[col] = winsorize_col(collapsed[col])
    after  = collapsed[col].describe()[["min", "max"]].to_dict()
    print(f"  {col}: [{before['min']:.1f}, {before['max']:.1f}] → [{after['min']:.1f}, {after['max']:.1f}]")

# log-transform skewed positive columns
collapsed["Log_Issue_Price"]   = np.log1p(collapsed["ISSUE PRICE"])
collapsed["Log_Latest_Revenue"] = np.log1p(collapsed["Latest_Revenue"].clip(lower=0))

# ── Step 8: impute remaining missing values ──────────────────────────────────

print("Step 8 — Imputing missing values...")
IMPUTE_COLS = [
    "Range_Width_pct", "Days_to_List", "Company_Age",
    "Avg_DE_pct", "Avg_Interest_Coverage", "Avg_ROCE_pct",
    "Avg_Assets_Turnover", "PAT_CAGR", "Revenue_CAGR",
    "Avg_Operating_Margin", "Avg_ROE_pct", "Log_Latest_Revenue",
]
for col in IMPUTE_COLS:
    n_missing = collapsed[col].isna().sum()
    if n_missing:
        med = collapsed[col].median()
        collapsed[col] = collapsed[col].fillna(med)
        print(f"  {col}: imputed {n_missing} missing with median {med:.3f}")

# ── Step 9: one-hot encode Assigned Industry ─────────────────────────────────

print("Step 9 — One-hot encoding Assigned Industry...")
industry_dummies = pd.get_dummies(
    collapsed["Assigned Industry"], prefix="Ind", drop_first=True
)
collapsed = pd.concat([collapsed, industry_dummies], axis=1)
print(f"  Industry dummies added: {list(industry_dummies.columns)}")

# ── Step 10: separate X (features) and Y (targets) ──────────────────────────

print("Step 10 — Separating feature matrix X and target Y...")
ind_dummy_cols = list(industry_dummies.columns)

FEATURE_COLS = [
    "Company_Age",
    "Log_Issue_Price",
    "Range_Width_pct",
    "Days_to_List",
    "Revenue_CAGR",
    "PAT_CAGR",
    "Avg_Operating_Margin",
    "Avg_ROE_pct",
    "Avg_ROCE_pct",
    "Avg_DE_pct",
    "Avg_Interest_Coverage",
    "Avg_Assets_Turnover",
    "Log_Latest_Revenue",
] + ind_dummy_cols

X = collapsed[["COMPANY NAME", "Symbol"] + FEATURE_COLS].copy()
Y = collapsed[["COMPANY NAME", "Symbol"] + TARGET_COLS].copy()

print(f"  X shape: {X.shape}")
print(f"  Y shape: {Y.shape}")
print(f"  NaNs in X features: {X[FEATURE_COLS].isna().sum().sum()}")

# ── Step 11: scale feature matrix ────────────────────────────────────────────

print("Step 11 — Scaling with RobustScaler...")
scaler = RobustScaler()
X_scaled_arr = scaler.fit_transform(X[FEATURE_COLS])
X_scaled = pd.DataFrame(X_scaled_arr, columns=FEATURE_COLS)
X_scaled.insert(0, "Symbol",       X["Symbol"].values)
X_scaled.insert(0, "COMPANY NAME", X["COMPANY NAME"].values)

# ── Save outputs ─────────────────────────────────────────────────────────────

print("\nSaving outputs...")
collapsed.to_excel("ipo_preprocessed.xlsx", index=False)
X_scaled.to_csv("ipo_features_scaled.csv", index=False)
Y.to_csv("ipo_targets.csv", index=False)

print(f"  ipo_preprocessed.xlsx  — {collapsed.shape[0]} rows × {collapsed.shape[1]} cols")
print(f"  ipo_features_scaled.csv — {X_scaled.shape[0]} rows × {X_scaled.shape[1]} cols")
print(f"  ipo_targets.csv         — {Y.shape[0]} rows × {Y.shape[1]} cols")
print("\nDone.")
