# IPO Research: Drivers of Indian IPO Listing Performance

Research pipeline investigating which factors — financial fundamentals, issue structure, or timing — drive listing performance for Indian IPOs.

**Status:** Active. Last updated 2026-06-23.

## Key finding

Across every method tried (correlation, clustering, random forest), **revenue growth is the only fundamental that consistently and significantly predicts IPO listing returns** (Spearman rho ≈ +0.22–0.26, p < 0.003 across 1d/30d/60d horizons). Profitability metrics (ROE, ROCE, operating margin, D/E, interest coverage) show ≈ zero relationship with listing returns, and fundamentals alone have essentially no predictive power (5-fold CV R² ≈ 0.001). This is consistent with the literature once you separate short-run listing pops (demand-driven — GMP, subscription ratios, none of which are yet in this dataset) from long-run operating performance (fundamentals-driven). See [Key findings](#key-findings) for full detail.

## Data sources

| File | Description |
|---|---|
| `data5years/*.xlsx` (10 files) | Raw financial export from ISI Markets. ~10,000 Indian companies × FY2022–FY2026, one row per company-year (revenue, profit, ROE, ROCE, D/E, interest coverage, asset turnover, margins, incorporation date, etc.). Most rows are unrelated listed/unlisted firms, not IPO companies. |
| `IPO data collection .xlsx` | Target list of 358 Indian IPO companies with IPO-specific fields (issue price, price band, listing returns at 1d/7d/15d/21d/30d/45d/60d, post-IPO price std, assigned industry). Several columns (GMP, subscription ratios, issue size, promoter holding) are empty placeholders, not yet sourced. |

## Pipeline

Run in this order:

1. **[match_companies.py](match_companies.py)** — Matches the 358 target IPO companies against the 10,000-company financial dataset (exact match, then fuzzy matching via `rapidfuzz`). Produces `company_matching_review.xlsx` for manual confirmation of fuzzy matches. **Result: 302 matched / 56 unmatched.**
2. **[merge_ipo_data.py](merge_ipo_data.py)** — Cross-joins all 358 IPO companies × 5 years, left-merges in financial columns keyed on `(Matched Financial Company, Year)`, preserving all 358 original rows (unmatched companies are NaN'd, not dropped). Produces `ipo_merged_dataset.xlsx` (1790 rows × 56 cols, long format).
3. **[preprocess.py](preprocess.py)** — Data-quality fixes (nullifies 5 confirmed-bad company-years, fixes one negative `Days_to_List`), derives per-company CAGR/margin/ROE/ROCE features across FY2022–26, winsorizes and scales. Produces `ipo_preprocessed.xlsx`, `ipo_features_scaled.csv`, `ipo_targets.csv`. Note: uses 2022–2026 averages, which pre-date real listing dates (superseded for driver analysis by step 5).
4. **[clustering.py](clustering.py)** — K-Means/Agglomerative clustering on the averaged features from step 3. Produces `ipo_clustered.xlsx`, `cluster_profiles.xlsx`, `cluster_performance.xlsx`, plus diagnostic charts.
5. **[preipo_driver_analysis.py](preipo_driver_analysis.py)** — Builds a true pre-IPO snapshot (last FY *before* listing, using sourced NSE listing dates) and runs the supervised driver analysis: Spearman correlation + Random Forest / permutation importance against listing returns. Produces `ipo_preipo_features.xlsx`, `preipo_feature_correlation.xlsx`, `preipo_feature_importance.xlsx`.
6. **[reclustering_preipo.py](reclustering_preipo.py)** — Re-runs clustering on the corrected pre-IPO snapshot from step 5 (fixes the look-ahead bias in step 4). Produces `ipo_reclustered_preipo.xlsx`, `recluster_profiles.xlsx`, `recluster_performance.xlsx`.

Reference/planning docs: [CLUSTERING_PLAN.md](CLUSTERING_PLAN.md), [ML_RESEARCH_ROADMAP.md](ML_RESEARCH_ROADMAP.md).

## Key findings

1. **Revenue growth is the only fundamental that consistently and significantly predicts IPO listing returns** (Spearman rho ≈ +0.22 to +0.26, p<0.003, across 1d/30d/60d horizons; also the top Random Forest feature by a wide margin).
2. **Profitability metrics (ROE, ROCE, Operating Margin, D/E, Interest Coverage) have ≈ zero relationship with listing returns** in this dataset — contradicting naive expectations from "fundamentals → IPO performance" literature, but consistent with the literature once you note that result is usually about *long-run* performance, not short-run listing pops.
3. **Clustering on fundamentals does not reliably separate listing returns** in either the original (issue-structure-driven, silhouette 0.48) or corrected pre-IPO (fundamental-quality-driven, silhouette 0.31) version. The corrected clustering does isolate a clean "mature/low-debt/high-margin" segment with directionally higher (and more volatile) returns, but the difference is not statistically significant (Kruskal-Wallis p>0.05 for all return horizons; only post-listing volatility differs significantly, p=0.037).
4. **Predictive power from fundamentals alone is essentially zero** (Random Forest 5-fold CV R² ≈ 0.001).
5. **Likely explanation:** short-run IPO listing returns in this market appear to be dominated by **demand-side dynamics** (grey market premium, subscription ratios, anchor investor interest) rather than company fundamentals — none of which are currently in the dataset. This is the single highest-value next step (see [ML_RESEARCH_ROADMAP.md](ML_RESEARCH_ROADMAP.md), Tier 1 data).

### Supporting detail

**Company matching (10,000 → 302 matched):** 277 exact matches + 25 fuzzy matches confirmed on manual review + 6 identified via known corporate renames (e.g. Zomato → Eternal Ltd., Adani Wilmar → AWL Agri Business, Macrotech Developers → Lodha Developers). 56 companies remain unmatched (`pending_companies.xlsx`, being sourced externally).

**Data-quality fixes:** A YoY revenue-ratio scan flagged 7 suspicious companies; 5 were web-verified as genuine data errors (wrong entity in source) and nullified for the bad year(s): Indiqube Spaces (2022), FiveStar Business Finance (2022 & 2023), BLS E-Services (2022), Jain Resource Recycling (2022), Sri Lotus Developers and Realty (2022). The remaining 2 (Solarworld Energy Solutions, Senores Pharmaceuticals) were confirmed as real growth, not errors.

**Clustering, Part 1 (pre-dates):** Best k=2, silhouette 0.483, split 53/249 companies. Driven mainly by `Range_Width_pct` (price-band width) and growth metrics, not profitability. Median Return_30d looked very different by cluster (1.2% vs 13.1%) but Kruskal-Wallis was not significant (p=0.21).

**Why clustering initially seemed to contradict fundamentals literature:** A direct Spearman correlation check confirmed this wasn't a bug — clustering finds groupings, not gradients, so it's the wrong tool to detect a monotonic fundamentals→returns relationship. This also surfaced a temporal-mismatch issue: FY2022–26 averages include post-listing years for many companies (look-ahead bias), motivating the sourcing of real listing dates.

**Real NSE listing dates:** Sourced for all 302 matched companies via web search, cross-verified against issue price. 300 verified; 2 flagged mismatches (Anupam Rasayan India — data shows ₹236 vs actual ₹555 IPO; Ruchi Soya Industries — data shows ₹34 vs actual ₹650, likely the 2022 Patanjali Foods FPO). Listing years span 2020–2025 (13/55/35/48/72/79 respectively). 231 of 302 companies have their pre-IPO fiscal year within the FY2022–26 data window.

**Driver analysis (pre-IPO snapshot, n=230):** See headline findings above. Model: Random Forest on Return_30d, 17 features (11 numeric + 6 industry dummies), n=192 after dropping missing targets.

**Re-clustering (pre-IPO snapshot, n=230):** Best k=2, silhouette 0.306 (lower than Part 1 — this cohort is more homogeneous once SME/fixed-price outliers are excluded). Produces a clean split: Cluster 0 (n=29, "mature/low-debt", D/E 15% vs 112%, interest coverage 59x vs 6x, age 28yrs vs 21yrs) vs Cluster 1 (n=201, "leveraged/growth-tilted"). `Range_Width_pct` and `Days_to_List` are now nearly identical across clusters, confirming the split reflects genuine fundamental quality rather than issue-structure artifacts. Returns are directionally higher for Cluster 0 across most horizons, but Kruskal-Wallis is not significant for any return metric (p=0.34–0.78); only post-IPO volatility differs significantly (p=0.037).

## Repository structure

**Scripts** (see [Pipeline](#pipeline) for run order and detail)
- `match_companies.py`, `merge_ipo_data.py`, `preprocess.py`, `clustering.py`, `preipo_driver_analysis.py`, `reclustering_preipo.py`

**Planning / reference docs**
- `CLUSTERING_PLAN.md` — clustering methodology decisions
- `ML_RESEARCH_ROADMAP.md` — catalog of applicable ML paradigms for future work, plus a 5-tier ranking of 17 candidate data points to source next (Tier 1: subscription ratios, GMP, issue size, anchor investor subscription)

**Data outputs**
- `company_matching_review.xlsx`, `pending_companies.xlsx`
- `ipo_merged_dataset.xlsx`
- `ipo_preprocessed.xlsx`, `ipo_features_scaled.csv`, `ipo_targets.csv`
- `ipo_listing_dates.xlsx` / `.csv`, `ipo_preprocessed_dated.xlsx`
- `ipo_preipo_features.xlsx`, `preipo_feature_correlation.xlsx`, `preipo_feature_importance.xlsx`
- `ipo_clustered.xlsx`, `cluster_profiles.xlsx`, `cluster_performance.xlsx`
- `ipo_reclustered_preipo.xlsx`, `recluster_profiles.xlsx`, `recluster_performance.xlsx`

**Charts**
- `cluster_k_selection.png`, `pca_projection.png`, `dendrogram.png`
- `preipo_correlation_heatmap.png`, `preipo_feature_importance.png`
- `reclustering_k_selection.png`, `reclustering_pca.png`

## Known limitations

- 56 of 358 IPO companies have no matched financial data (sourced separately).
- 71 companies (mostly listed 2020–21) predate the FY2022–26 financial data window and are excluded from the pre-IPO-snapshot analyses — those run on ~230/302 companies, not the full set.
- 5 very recent listings lack post-listing return data yet.
- 2 companies have unresolved listing-date/issue-price mismatches (Anupam Rasayan India, Ruchi Soya Industries).
- No demand-side data (GMP, subscription ratios, issue size, promoter holding, underwriter identity) — believed to be the primary driver of listing-day pops; see `ML_RESEARCH_ROADMAP.md` Tier 1.

## Suggested next steps

Per `ML_RESEARCH_ROADMAP.md`: (1) OLS + Lasso + fixed-effects for signed drivers, (2) XGBoost + SHAP, (3) binary win/loss classification, (4) return-trajectory clustering + GMM/HDBSCAN, (5) t-SNE/UMAP visualization, (6) market-cycle/hot-cold window analysis, (7) source Tier-1 demand-side data and re-run steps 1–3, (8) advanced causal/RL/NLP methods.
