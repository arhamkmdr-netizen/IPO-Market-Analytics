# IPO Research — ML/Analytics Roadmap

A living catalog of analyses, algorithms, and models we can run on this dataset,
organized by ML paradigm and research purpose. We'll implement these one by one.

**Legend (feasibility with data we currently have):**
- 🟢 Ready now — runnable on current data
- 🟡 Partial — runnable but weak/limited until more data sourced
- 🔴 Blocked — needs additional data first (see Section 9)

**What we currently have (302 matched companies):**
pre-IPO financials (FY2022-26: revenue, profit, ROE, ROCE, D/E, interest coverage,
asset turnover, margins), issue price, price-band width, days-to-list, listing returns
(1d/7d/15d/21d/30d/45d/60d), post-IPO price volatility, industry (7 sectors),
listing dates, company age. 56 companies pending financials.

---

## 0. Status so far
- ✅ K-Means + Hierarchical clustering (clusters split on issue structure, not returns)
- ✅ Spearman correlation + Random Forest driver analysis on pre-IPO snapshot
- **Headline:** revenue growth is the only fundamental linked to listing returns;
  profitability ≈ 0; returns largely unpredictable from fundamentals (R²≈0) →
  demand-side factors likely dominate.

---

## 1. SUPERVISED — Regression (predict return magnitude)
Target = Return_1d / 30d / 60d, or listing gain %.

| # | Model | Purpose / insight | Feasibility |
|---|---|---|---|
| 1.1 | **OLS / Linear regression** (with robust SE) | Signed, interpretable effect sizes ("+1pp revenue growth → +X% return"); baseline | 🟢 |
| 1.2 | **Regularized regression** (Ridge / Lasso / ElasticNet) | Feature selection, handle multicollinearity, see which drivers survive shrinkage | 🟢 |
| 1.3 | **Quantile regression** | Model the *tails* — what drives the best vs worst listings, not just the mean | 🟢 |
| 1.4 | **Gradient boosting** (XGBoost / LightGBM / CatBoost) | Best-in-class tabular accuracy; non-linearities + interactions | 🟢 |
| 1.5 | **Random Forest** (done) | Importance ranking baseline | ✅ |
| 1.6 | **SHAP analysis** on the boosting model | Per-feature, per-company contribution; direction + interaction effects (the rigorous "what affects price" answer) | 🟢 |
| 1.7 | **Generalized Additive Models (GAM)** | Capture non-linear feature→return shapes while staying interpretable | 🟢 |
| 1.8 | **Multi-output regression** | Predict the whole return curve (1d→60d) jointly | 🟢 |

## 2. SUPERVISED — Classification
Convert returns to categories.

| # | Model | Purpose / insight | Feasibility |
|---|---|---|---|
| 2.1 | **Binary: positive vs negative listing** (logistic regression) | Odds-ratio drivers of a "winning" listing; clean interpretation | 🟢 |
| 2.2 | **Multiclass: loss / flat / moderate / blockbuster** (>50%) | Which profiles predict each outcome bucket | 🟢 |
| 2.3 | **Tree ensembles for classification** (RF/XGBoost) + SHAP | Non-linear classifier + explainability | 🟢 |
| 2.4 | **Calibration analysis** | Are predicted "win" probabilities reliable? (useful if used for decisions) | 🟢 |
| 2.5 | **"Stag profit" classifier** — predicts if 1d return > threshold | Practical retail question: will it pop on listing day? | 🟡 (better with GMP/subscription) |

## 3. UNSUPERVISED — Clustering & structure
| # | Model | Purpose / insight | Feasibility |
|---|---|---|---|
| 3.1 | K-Means / Hierarchical (done) | Natural company segments | ✅ |
| 3.2 | **Gaussian Mixture Models** | Soft clustering + probabilistic membership; handles elliptical clusters | 🟢 |
| 3.3 | **DBSCAN / HDBSCAN** | Density-based; isolates outlier IPOs (unusual profiles) | 🟢 |
| 3.4 | **K-Prototypes** | Mixed numeric+categorical clustering with industry native | 🟢 |
| 3.5 | **Cluster on RETURN trajectories** (1d→60d shapes) | Group by post-listing price *behavior* (pop-and-fade, steady climb, sink) then explain with fundamentals | 🟢 |

## 4. UNSUPERVISED — Dimensionality reduction & visualization
| # | Model | Purpose / insight | Feasibility |
|---|---|---|---|
| 4.1 | PCA (done in clustering) | Variance structure, decorrelation | ✅ |
| 4.2 | **t-SNE / UMAP** | 2D maps revealing non-linear groupings; color by return to spot pockets | 🟢 |
| 4.3 | **Factor Analysis** | Latent factors behind correlated financials (e.g., "profitability factor", "leverage factor") | 🟢 |

## 5. UNSUPERVISED — Anomaly detection & association
| # | Model | Purpose / insight | Feasibility |
|---|---|---|---|
| 5.1 | **Isolation Forest / LOF** | Flag atypical IPOs (data-quality + genuinely unusual deals) | 🟢 |
| 5.2 | **Association rule mining** (Apriori/FP-Growth) | "IF high revenue growth AND book-built AND tech → THEN strong listing" rules on binned features | 🟢 |

## 6. CAUSAL INFERENCE / ECONOMETRICS (best fit for "what *affects* price")
Correlation ≠ causation; these target the research question directly.

| # | Method | Purpose / insight | Feasibility |
|---|---|---|---|
| 6.1 | **Multiple regression with controls** (sector, year FE) | Isolate a factor's effect holding others constant | 🟢 |
| 6.2 | **Fixed-effects (industry & listing-year)** | Remove sector/market-cycle confounding | 🟢 |
| 6.3 | **Double ML / causal forests** | Modern ML-based causal effect estimates of each driver | 🟡 |
| 6.4 | **Mediation analysis** | Does revenue growth act *through* subscription/GMP? | 🔴 (needs GMP/subscription) |
| 6.5 | **Propensity / matching** (book-built vs fixed-price, etc.) | Treatment-effect style comparison of IPO structures | 🟡 |

## 7. TIME-SERIES / SEQUENCE
| # | Method | Purpose / insight | Feasibility |
|---|---|---|---|
| 7.1 | **IPO-volume / market-cycle analysis** | Do hot/cold IPO windows predict returns? (we have listing dates → monthly cohorts) | 🟢 |
| 7.2 | **Post-listing price-path modeling** (if we pull daily prices) | Model the 60-day trajectory; momentum vs mean-reversion | 🔴 (needs daily price series) |
| 7.3 | **Sequence models (LSTM/Temporal)** on multi-year financials | Learn financial *trajectory* → return, not just snapshot | 🟡 (only ~5 yrs/firm) |

## 8. REINFORCEMENT LEARNING & optimization
| # | Method | Purpose / insight | Feasibility |
|---|---|---|---|
| 8.1 | **Bandit / RL "IPO application" policy** | Learn an allocation policy: which IPOs to apply for to maximize listing-day return under a budget; reward = realized return | 🟡 (works on historical replay; richer with GMP/subscription) |
| 8.2 | **Portfolio optimization** (mean-variance / CVaR) | Optimal capital allocation across an IPO cohort given predicted return + volatility | 🟢 |
| 8.3 | **RL exit-timing agent** (hold 1d…60d) | When to sell post-listing to maximize return | 🔴 (needs daily prices) |

## 9. NLP (requires text data — see Section 11)
| # | Method | Purpose / insight | Feasibility |
|---|---|---|---|
| 9.1 | **RHP/DRHP risk-factor analysis** | Length, sentiment, risk-count of prospectus → returns (well-studied signal) | 🔴 |
| 9.2 | **News/social sentiment around listing** | Pre-listing buzz → pop | 🔴 |
| 9.3 | **Topic modeling on business descriptions** | Data-driven sector/theme tags beyond the 7 industries | 🟡 (need descriptions) |

## 10. GRAPH / NETWORK & SURVIVAL
| # | Method | Purpose / insight | Feasibility |
|---|---|---|---|
| 10.1 | **Underwriter–IPO bipartite network** | Lead-manager reputation effect on returns (classic finding) | 🔴 (need underwriter data) |
| 10.2 | **Survival analysis** | Time-to-fall-below-issue-price; which factors prolong gains | 🔴 (needs daily prices) |

---

## 11. ADDITIONAL DATA THAT WOULD UNLOCK THE MOST
Ranked by expected research value. Items 1-4 are the empty columns already in
`IPO data collection .xlsx` — highest priority since the analysis points to demand-side drivers.

**Tier 1 — demand-side (most likely the real drivers of listing pops):**
1. **Subscription ratios** (QIB / NII-HNI / Retail / Overall) — strongest known predictor of listing-day return.
2. **Grey Market Premium (GMP)** — pre-listing demand proxy; often near-deterministic of the pop.
3. **Issue size** + fresh-issue vs offer-for-sale split — size/liquidity effects.
4. **Anchor investor subscription / quality** — institutional validation signal.

**Tier 2 — valuation & ownership:**
5. **P/E, P/B, EV/EBITDA at issue price** vs sector median — over/under-pricing.
6. **Promoter holding pre & post-issue** + dilution %.
7. **Price band revisions** / whether priced at top of band.

**Tier 3 — market context (per listing date — we have dates, so joinable):**
8. **Nifty/Sensex return** in the 2-4 weeks before listing + on listing day.
9. **India VIX** (volatility regime) at listing.
10. **Sector index return** around listing.
11. **FII/DII net flows** in listing month.

**Tier 4 — intermediary & structure:**
12. **Lead manager / underwriter** identity (reputation effect).
13. **IPO mechanism**: book-built vs fixed price; mainboard vs SME.
14. **IPO rating** (CRISIL etc.), **lock-in expiry** dates.

**Tier 5 — outcome enrichment:**
15. **Daily post-listing price series** (unlocks trajectory modeling, survival, RL exit-timing, long-run returns).
16. **Long-run returns** (6-month, 1-year, 3-year) — to test the *fundamentals→long-run performance* hypothesis (where profitability/EBITDA should matter, per the literature).
17. **Listing-day high/low/close** (intraday volatility, not just std).

---

## Suggested implementation order
1. **OLS + Lasso + fixed effects (6.1, 6.2, 1.1, 1.2)** — interpretable signed drivers; quick win, directly answers research question.
2. **XGBoost + SHAP (1.4, 1.6)** — best predictive model + rigorous explainability.
3. **Binary win/loss classification + logistic (2.1)** — clean odds-ratio story.
4. **Return-trajectory clustering (3.5) + GMM/HDBSCAN (3.2, 3.3)** — behavioral segments.
5. **t-SNE/UMAP (4.2)** — visualization for the writeup.
6. **Market-cycle / hot-cold window analysis (7.1)** — uses listing dates we sourced.
7. **Source Tier-1 demand-side data**, then re-run 1-3 (expected big jump in R²).
8. Advanced: causal forests (6.3), RL application policy (8.1), NLP on RHPs (9.x).
