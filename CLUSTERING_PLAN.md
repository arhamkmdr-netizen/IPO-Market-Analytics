# IPO Research — Clustering & Driver Analysis Plan

## Context
We have a clean, scaled feature matrix (`ipo_features_scaled.csv`, 302 companies × 21 cols)
and outcome variables (`ipo_targets.csv`, returns + post-IPO volatility).
Goal: (A) segment IPO companies by their pre-IPO fundamentals + issue characteristics,
and (B) directly identify which factors drive IPO listing performance.

We will run **two separate analyses**:
1. **Unsupervised clustering** — find natural company segments, then see how each
   segment performed post-listing.
2. **Supervised driver analysis** — directly rank which features predict returns.

---

## Decisions locked in
- **Industry**: NOT a clustering input. Cluster on financial + issue features only;
  use `Assigned Industry` as a profiling/interpretation lens afterward.
- **Algorithms**: Compare **K-Means** and **Agglomerative (hierarchical)**; pick by
  silhouette score + interpretability.
- **Driver analysis**: Run separately (correlation + tree-based feature importance).

---

# PART 1 — CLUSTERING

## Step 1 — Finalize the feature matrix
The current scaled file has two features whose variance dominates everything else
(Range_Width_pct var≈234, Days_to_List var≈573 vs ~0.5–5 for the rest) because of
extreme outliers RobustScaler couldn't tame.

- Re-run feature prep: winsorize `Range_Width_pct` and `Days_to_List` (5th–95th pct)
  **before** scaling, so no single feature dominates K-Means distance.
- Keep both `Avg_ROE_pct` and `Avg_ROCE_pct` (r=0.75 — correlated but each adds signal);
  note the redundancy in interpretation.
- **Drop industry dummies** from the clustering feature set (per decision).
- Final clustering features (~13 continuous):
  `Company_Age, Log_Issue_Price, Range_Width_pct, Days_to_List, Revenue_CAGR,
  PAT_CAGR, Avg_Operating_Margin, Avg_ROE_pct, Avg_ROCE_pct, Avg_DE_pct,
  Avg_Interest_Coverage, Avg_Assets_Turnover, Log_Latest_Revenue`
- Output: `ipo_features_for_clustering.csv` (re-scaled with RobustScaler).

## Step 2 — Determine optimal k
- Compute across k = 2..10:
  - Inertia (elbow method)
  - Silhouette score
  - Calinski-Harabasz index
  - Davies-Bouldin index
- Plot all four; choose candidate k (likely 3–5).
- Save: `cluster_k_selection.png`.

## Step 3 — PCA projection (visualization only)
- Fit PCA, keep 2 components for a 2D scatter to eyeball separation.
- Report variance explained by PC1/PC2.
- Clustering itself runs on full feature space, not PCA components.
- Save: `pca_projection.png`.

## Step 4 — Fit & compare algorithms
- **K-Means** at chosen k (n_init=10, random_state=42).
- **Agglomerative** (Ward linkage) at same k; also produce a dendrogram.
- Compare silhouette scores; pick the primary model.
- Save: `dendrogram.png`.

## Step 5 — Profile clusters
For the chosen model:
- Mean (un-scaled) feature values per cluster → build a profile table.
- Assign descriptive names (e.g. "mature profitable", "high-growth loss-makers").
- Industry composition per cluster (cross-tab).
- Save: `cluster_profiles.xlsx`.

## Step 6 — Link clusters to IPO performance
- Merge cluster labels with `ipo_targets.csv`.
- Mean/median Return_1d / 30d / 60d and PostIPO_price_std per cluster.
- **Kruskal-Wallis test** per return horizon: do returns differ significantly
  across clusters? (Non-parametric; returns are skewed.)
- Interpret: which company profiles tend to list well / poorly / volatile.
- Save: `cluster_performance.xlsx`.

---

# PART 2 — SUPERVISED DRIVER ANALYSIS (separate)

Goal: directly rank which factors affect IPO performance, independent of clustering.

## Step 7 — Correlation with returns
- Spearman correlation (robust to non-linearity/outliers) of each feature against
  `Return_1d`, `Return_30d`, `Return_60d`.
- Rank features by |correlation|; flag statistically significant ones (p<0.05).
- Save: `feature_return_correlation.xlsx` + heatmap `correlation_heatmap.png`.

## Step 8 — Tree-based feature importance
- Target: `Return_30d` (primary), repeat for `Return_1d` as a robustness check.
- Model: Random Forest regressor (handles non-linearity + interactions).
- Report feature importances + permutation importance (more reliable).
- Include `Assigned Industry` here (one-hot) since supervised models handle it fine.
- Note: R² will likely be modest — IPO returns are noisy and many drivers
  (GMP, subscription ratios) are still missing. Focus on *relative* importance ranking.
- Save: `feature_importance.png` + `feature_importance.xlsx`.

## Step 9 — Synthesis
- Combine: which features show up as both (a) cluster-differentiating AND
  (b) return-predictive → strongest candidate drivers of IPO performance.
- Short written summary of findings.

---

## Caveats to keep in mind
- 297/302 companies have return data (5 very recent listings lack it) — excluded
  from performance steps, retained for clustering.
- Many theorized IPO drivers (GMP, QIB/retail subscription, issue size, promoter
  holding) are not yet in the data — findings are limited to fundamentals + issue
  price/band characteristics until the 56 pending companies + extra columns are sourced.
- This is an associational analysis, not causal.

## Output files summary
| File | Contents |
|---|---|
| `ipo_features_for_clustering.csv` | Re-scaled 13-feature matrix |
| `cluster_k_selection.png` | Elbow / silhouette / CH / DB plots |
| `pca_projection.png` | 2D PCA scatter colored by cluster |
| `dendrogram.png` | Hierarchical clustering tree |
| `cluster_profiles.xlsx` | Per-cluster feature means + industry mix |
| `cluster_performance.xlsx` | Per-cluster returns + Kruskal-Wallis results |
| `feature_return_correlation.xlsx` + `correlation_heatmap.png` | Spearman drivers |
| `feature_importance.xlsx` + `feature_importance.png` | RF / permutation importance |
