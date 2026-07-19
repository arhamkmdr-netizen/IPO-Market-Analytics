"""
Support Vector Regression (SVR) vs other regressors — predicting listing returns.
(Roadmap items 1.1, 1.2, 1.4 + SVR)

Uses the pre-IPO snapshot cohort (same features as preipo_driver_analysis.py).
Compares, with nested-free but honest k-fold cross-validation:
  - SVR (RBF kernel)  + SVR (linear kernel)
  - OLS / Linear
  - Ridge, Lasso
  - Random Forest
  - Gradient Boosting
  - Baseline: predict-the-mean (DummyRegressor)

Scale-sensitive models (SVR, linear family) are wrapped in a Pipeline that
RobustScales inside each CV fold (no leakage). Tree models use raw features.

Metrics per model per horizon (5-fold CV): R2, MAE, RMSE.

Outputs:
  svr_model_comparison.xlsx   - full metric table (all models x horizons)
  svr_model_comparison.png    - CV R2 bar chart per horizon
  svr_pred_vs_actual.png      - best model predicted vs actual (Return_30d)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.model_selection import KFold, cross_val_predict, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVR

RANDOM_STATE = 42

FEATURES = [
    "PreIPO_ROE_pct", "PreIPO_ROCE_pct", "PreIPO_Operating_Margin",
    "PreIPO_Revenue_Growth", "PreIPO_PAT_Growth", "PreIPO_DE_pct",
    "PreIPO_Interest_Coverage", "PreIPO_Assets_Turnover",
    "Log_PreIPO_Revenue", "Log_Issue_Price", "Company_Age_at_IPO",
]
HORIZONS = ["Return_1d", "Return_30d", "Return_60d"]


def winsorize_clip(s, lo=0.05, hi=0.95):
    return s.clip(s.quantile(lo), s.quantile(hi))


# ── Load features ────────────────────────────────────────────────────────────
print("Loading pre-IPO features...")
feat = pd.read_excel("ipo_preipo_features.xlsx")

# median-impute the numeric predictors (same as clustering scripts)
for col in FEATURES:
    if feat[col].isna().any():
        feat[col] = feat[col].fillna(feat[col].median())

# winsorize heavy-tailed predictors
for col in ["PreIPO_ROE_pct", "PreIPO_ROCE_pct", "PreIPO_DE_pct",
            "PreIPO_Interest_Coverage", "PreIPO_Revenue_Growth", "PreIPO_PAT_Growth"]:
    feat[col] = winsorize_clip(feat[col])

ind_dummies = pd.get_dummies(feat["Assigned Industry"], prefix="Ind", drop_first=True)
X_all = pd.concat([feat[FEATURES], ind_dummies], axis=1)

# ── Define models ────────────────────────────────────────────────────────────
def build_models():
    return {
        "Baseline (mean)":   DummyRegressor(strategy="mean"),
        "OLS":               make_pipeline(RobustScaler(), LinearRegression()),
        "Ridge":             make_pipeline(RobustScaler(), Ridge(alpha=10.0, random_state=RANDOM_STATE)),
        "Lasso":             make_pipeline(RobustScaler(), Lasso(alpha=0.5, random_state=RANDOM_STATE, max_iter=5000)),
        "SVR (RBF)":         make_pipeline(RobustScaler(), SVR(kernel="rbf", C=10.0, gamma="scale", epsilon=1.0)),
        "SVR (linear)":      make_pipeline(RobustScaler(), SVR(kernel="linear", C=1.0, epsilon=1.0)),
        "Random Forest":     RandomForestRegressor(n_estimators=400, max_depth=6,
                                                   min_samples_leaf=5, random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=300, max_depth=3,
                                                       learning_rate=0.03, random_state=RANDOM_STATE),
    }


cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
rows = []
best_overall = {"r2": -np.inf}

for horizon in HORIZONS:
    df = pd.concat([X_all, feat[horizon]], axis=1).dropna()
    X, y = df[X_all.columns], df[horizon]
    print(f"\n=== {horizon}  (n={len(df)}, {X.shape[1]} features) ===")
    for name, model in build_models().items():
        r2 = cross_val_score(model, X, y, cv=cv, scoring="r2")
        mae = -cross_val_score(model, X, y, cv=cv, scoring="neg_mean_absolute_error")
        rmse = -cross_val_score(model, X, y, cv=cv, scoring="neg_root_mean_squared_error")
        rows.append({
            "Horizon": horizon, "Model": name, "N": len(df),
            "CV_R2_mean": round(r2.mean(), 4), "CV_R2_std": round(r2.std(), 4),
            "CV_MAE": round(mae.mean(), 2), "CV_RMSE": round(rmse.mean(), 2),
        })
        print(f"  {name:20s} R2={r2.mean():+.4f} (+/-{r2.std():.3f})  MAE={mae.mean():.2f}  RMSE={rmse.mean():.2f}")
        if horizon == "Return_30d" and r2.mean() > best_overall["r2"] and name != "Baseline (mean)":
            best_overall = {"r2": r2.mean(), "name": name, "model": model, "X": X, "y": y}

results = pd.DataFrame(rows)
results.to_excel("svr_model_comparison.xlsx", index=False)

# ── Bar chart of CV R2 per horizon ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))
models_order = list(build_models().keys())
width = 0.25
x = np.arange(len(models_order))
for i, horizon in enumerate(HORIZONS):
    sub = results[results["Horizon"] == horizon].set_index("Model").loc[models_order]
    ax.bar(x + i * width, sub["CV_R2_mean"], width, label=horizon)
ax.axhline(0, color="black", lw=0.8)
ax.set_xticks(x + width)
ax.set_xticklabels(models_order, rotation=30, ha="right")
ax.set_ylabel("5-fold CV R²")
ax.set_title("Model comparison — CV R² by return horizon (higher = better; 0 = no better than mean)")
ax.legend()
plt.tight_layout(); plt.savefig("svr_model_comparison.png", dpi=120); plt.close()

# ── Predicted vs actual for the best Return_30d model ────────────────────────
if "model" in best_overall:
    pred = cross_val_predict(best_overall["model"], best_overall["X"], best_overall["y"], cv=cv)
    plt.figure(figsize=(7, 7))
    plt.scatter(best_overall["y"], pred, alpha=0.6, edgecolors="k", linewidths=0.3)
    lims = [min(best_overall["y"].min(), pred.min()), max(best_overall["y"].max(), pred.max())]
    plt.plot(lims, lims, "r--", label="perfect prediction")
    plt.xlabel("Actual Return_30d (%)"); plt.ylabel("Predicted Return_30d (%)")
    plt.title(f"Best model on Return_30d: {best_overall['name']} "
              f"(CV R²={best_overall['r2']:.3f})")
    plt.legend(); plt.tight_layout(); plt.savefig("svr_pred_vs_actual.png", dpi=120); plt.close()

print("\nBest Return_30d model:", best_overall.get("name"), f"(R2={best_overall['r2']:.3f})")
print("Saved: svr_model_comparison.xlsx, svr_model_comparison.png, svr_pred_vs_actual.png")
