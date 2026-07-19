"""
Filter the combined financial dataset (data5years/*.xlsx) down to only the
companies confirmed as IPO matches in company_matching_review.xlsx.

Output: ipo_financials_filtered.xlsx
  - One row per (IPO company, year) -- 5 years per company
  - 'IPO Company Name' column added so this can be joined back to the
    IPO data collection file later.
"""

import glob

import pandas as pd

DATA_DIR = "data5years"
REVIEW_FILE = "company_matching_review.xlsx"
OUTPUT_FILE = "ipo_financials_filtered.xlsx"


def main():
    # 1. Combine all financial export files
    files = sorted(glob.glob(f"{DATA_DIR}/*.xlsx"))
    fin = pd.concat([pd.read_excel(f, header=7) for f in files], ignore_index=True)

    # 2. Load confirmed mapping
    review = pd.read_excel(REVIEW_FILE)
    confirmed = review[review["Confirm (y/n)"] == "y"].copy()

    # Sanity checks
    dup_targets = confirmed["Matched Financial Company"].value_counts()
    dup_targets = dup_targets[dup_targets > 1]
    if not dup_targets.empty:
        print("WARNING: multiple IPO companies map to the same financial company:")
        print(dup_targets)

    mapping = dict(zip(confirmed["Matched Financial Company"], confirmed["IPO Company Name"]))

    # 3. Filter financial data to matched companies
    filtered = fin[fin["Company"].isin(mapping.keys())].copy()
    filtered.insert(0, "IPO Company Name", filtered["Company"].map(mapping))

    # 4. Check coverage
    matched_companies = set(mapping.keys())
    found_companies = set(filtered["Company"].unique())
    missing = matched_companies - found_companies

    filtered = filtered.sort_values(["IPO Company Name", "Year"]).reset_index(drop=True)
    filtered.to_excel(OUTPUT_FILE, index=False)

    print(f"Confirmed mappings: {len(confirmed)}")
    print(f"Companies found in financial data: {len(found_companies)}")
    print(f"Rows in filtered dataset: {len(filtered)} (expected {len(found_companies) * 5})")
    if missing:
        print(f"WARNING: {len(missing)} confirmed companies not found in financial data: {missing}")
    print(f"\nWrote {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
