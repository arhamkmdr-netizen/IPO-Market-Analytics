"""
Merge the IPO data collection file with the financial data (data5years/*.xlsx),
using the confirmed company name mapping from company_matching_review.xlsx.

Output: ipo_merged_dataset.xlsx
  - Long format: one row per (IPO company, year) -> 358 companies x 5 years = 1790 rows
  - All original IPO data collection columns (incl. Symbol) are repeated across
    each company's 5 yearly rows
  - Financial columns are filled in for the 302 matched companies; the 56
    unmatched companies have NaN financial columns (placeholder for later)
  - 'Matched Financial Company' column added for traceability
"""

import glob

import pandas as pd

DATA_DIR = "data5years"
IPO_FILE = "IPO data collection .xlsx"
REVIEW_FILE = "company_matching_review.xlsx"
OUTPUT_FILE = "ipo_merged_dataset.xlsx"

YEARS = [2022, 2023, 2024, 2025, 2026]


def main():
    # 1. Load IPO data collection (358 companies, 1 row each)
    ipo = pd.read_excel(IPO_FILE, sheet_name="IPO_FINAL_CLEANED_FOR_PLOTTING")

    # 2. Load confirmed mapping (IPO Company Name -> Matched Financial Company)
    review = pd.read_excel(REVIEW_FILE)
    confirmed = review[review["Confirm (y/n)"] == "y"]
    mapping = dict(zip(confirmed["IPO Company Name"], confirmed["Matched Financial Company"]))

    ipo["Matched Financial Company"] = ipo["COMPANY NAME"].map(mapping)

    # 3. Load combined financial data (10,000 companies x 5 years)
    files = sorted(glob.glob(f"{DATA_DIR}/*.xlsx"))
    fin = pd.concat([pd.read_excel(f, header=7) for f in files], ignore_index=True)

    # Financial columns to bring over (exclude identifiers we already have / don't need duplicated)
    fin_cols = [
        "Total Operating Revenue", "Operating Profit", "Profit before Income Tax",
        "Net Profit/Loss for the Period", "Debt / Equity (%)", "Interest Coverage Ratio (x)",
        "Assets Turnover (x)", "Return on Equity (ROE) (%)", "Price / Earnings (P/E) (x)",
        "Incorporation Date", "Return on Capital Employed (%)", "Fiscal Year",
        "Audited", "Consolidated", "Source",
    ]
    fin_subset = fin[["Company", "Year"] + fin_cols]

    # 4. Build long format: 358 companies x 5 years
    ipo["_key"] = 1
    years_df = pd.DataFrame({"Year": YEARS})
    years_df["_key"] = 1
    long_df = ipo.merge(years_df, on="_key").drop(columns="_key")

    # 5. Merge in financial data on (Matched Financial Company, Year)
    long_df = long_df.merge(
        fin_subset,
        left_on=["Matched Financial Company", "Year"],
        right_on=["Company", "Year"],
        how="left",
    ).drop(columns="Company")

    long_df = long_df.sort_values(["COMPANY NAME", "Year"]).reset_index(drop=True)
    long_df.to_excel(OUTPUT_FILE, index=False)

    n_companies = ipo["COMPANY NAME"].nunique()
    n_matched = ipo["Matched Financial Company"].notna().sum()
    print(f"IPO companies: {n_companies}")
    print(f"Matched to financial data: {n_matched}")
    print(f"Output rows: {len(long_df)} (expected {n_companies * len(YEARS)})")
    print(f"\nWrote {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
