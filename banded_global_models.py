"""
Global model comparison across TARGET BANDING schemes.

We hold the (pooled) feature set fixed and vary how the Return_30d target is
discretized, to see which granularity is actually learnable:

  - Binary : neg (<0) / pos (>=0)
  - 3-band : loss (<0) / modest (0-25) / strong (>=25)
  - 5-band : neg / 0-10 / 10-25 / 25-50 / 50+
  - (reference) Regression on raw return %

Accuracy is NOT comparable across schemes (different # classes / baselines), so the
headline metric is macro one-vs-rest AUC (0.5 = coin flip, comparable across schemes)
plus accuracy LIFT over the majority-class baseline and macro-F1.

Models: multinomial Logistic + Random Forest, 5-fold stratified CV.

Outputs:
  banded_global_models.xlsx   - metric table (scheme x model)
  banded_global_models.png    - macro-AUC and accuracy-lift bars
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_predict, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

RANDOM_STATE = 42

DRIVERS = ["PreIPO_Revenue_Growth", "PreIPO_PAT_Growth", "PreIPO_ROE_pct",
           "PreIPO_ROCE_pct", "PreIPO_Operating_Margin", "PreIPO_DE_pct",
           "PreIPO_Interest_Coverage", "PreIPO_Assets_Turnover",
           "Log_PreIPO_Revenue", "Company_Age_at_IPO", "Log_Issue_Price"]
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

ind_dum = pd.get_dummies(feat["Assigned Industry"], prefix="Ind", drop_first=True)
X = pd.concat([feat[DRIVERS], ind_dum], axis=1).astype(float).values
r = feat[TARGET].values

# ── band definitions ─────────────────────────────────────────────────────────
def to_binary(x):  return (x >= 0).astype(int)
def to_3band(x):
    return np.select([x < 0, x < 25], [0, 1], default=2)
def to_5band(x):
    return np.select([x < 0, x < 10, x < 25, x < 50], [0, 1, 2, 3], default=4)

SCHEMES = {
    "Binary (neg/pos)":            to_binary(r),
    "3-band (loss/modest/strong)": to_3band(r),
    "5-band":                      to_5band(r),
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


def macro_auc(model, X, y):
    """OvR macro AUC via cross-validated probabilities (comparable across schemes)."""
    proba = cross_val_predict(model, X, y, cv=skf, method="predict_proba")
    classes = np.unique(y)
    if len(classes) == 2:
        return roc_auc_score(y, proba[:, 1])
    return roc_auc_score(y, proba, multi_class="ovr", average="macro", labels=classes)


rows = []
for scheme, y in SCHEMES.items():
    y = np.asarray(y)
    n_classes = len(np.unique(y))
    baseline_acc = pd.Series(y).value_counts(normalize=True).max()  # majority class

    for mname, model in {
        "Logistic": make_pipeline(RobustScaler(),
                                  LogisticRegression(max_iter=2000, C=0.5)),
        "RandomForest": RandomForestClassifier(n_estimators=400, max_depth=6,
                                               min_samples_leaf=5, random_state=RANDOM_STATE),
    }.items():
        acc = cross_val_score(model, X, y, cv=skf, scoring="accuracy").mean()
        pred = cross_val_predict(model, X, y, cv=skf)
        f1m = f1_score(y, pred, average="macro")
        auc = macro_auc(model, X, y)
        rows.append({
            "Scheme": scheme, "Model": mname, "N_classes": n_classes,
            "Baseline_acc": round(baseline_acc, 3),
            "CV_acc": round(acc, 3),
            "Acc_lift": round(acc - baseline_acc, 3),
            "Macro_F1": round(f1m, 3),
            "Macro_AUC": round(auc, 3),
        })
        print(f"  {scheme:30s} {mname:13s} acc={acc:.3f} (base {baseline_acc:.3f}, "
              f"lift {acc-baseline_acc:+.3f})  F1={f1m:.3f}  AUC={auc:.3f}")

# reference: regression on raw returns
reg = RandomForestRegressor(n_estimators=400, max_depth=6, min_samples_leaf=5,
                            random_state=RANDOM_STATE)
reg_r2 = cross_val_score(reg, X, r, cv=kf, scoring="r2").mean()
print(f"\n  Reference — Regression on raw return %: CV R2={reg_r2:.3f}")

res = pd.DataFrame(rows)
res.to_excel("banded_global_models.xlsx", index=False)

# ── plot ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(1, 2, figsize=(14, 6))
piv_auc = res.pivot(index="Scheme", columns="Model", values="Macro_AUC")
piv_lift = res.pivot(index="Scheme", columns="Model", values="Acc_lift")
order = ["Binary (neg/pos)", "3-band (loss/modest/strong)", "5-band"]
piv_auc = piv_auc.loc[order]; piv_lift = piv_lift.loc[order]

piv_auc.plot(kind="bar", ax=ax[0], rot=15)
ax[0].axhline(0.5, color="red", ls="--", label="coin flip (0.5)")
ax[0].set_title("Macro one-vs-rest AUC (comparable across schemes)")
ax[0].set_ylabel("AUC"); ax[0].legend()

piv_lift.plot(kind="bar", ax=ax[1], rot=15)
ax[1].axhline(0.0, color="red", ls="--", label="no better than majority")
ax[1].set_title("Accuracy lift over majority-class baseline")
ax[1].set_ylabel("CV accuracy − baseline"); ax[1].legend()
plt.tight_layout(); plt.savefig("banded_global_models.png", dpi=120); plt.close()

print("\nSaved: banded_global_models.xlsx, banded_global_models.png")
