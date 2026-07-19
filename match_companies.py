"""
Match IPO company names (IPO data collection .xlsx) against the combined
financial dataset (data5years/*.xlsx) and produce a review spreadsheet.

Output: company_matching_review.xlsx
  - IPO Company Name
  - Match Type: "exact_normalized", "fuzzy", or "no_match"
  - Matched Financial Company (best candidate)
  - Match Score (0-100)
  - Alt Match 2 / Alt Score 2 (second-best candidate, for context)
  - Confirm (blank column for manual review: y/n)
"""

import glob
import re

import pandas as pd
from rapidfuzz import fuzz, process

DATA_DIR = "data5years"
IPO_FILE = "IPO data collection .xlsx"
OUTPUT_FILE = "company_matching_review.xlsx"

SUFFIX_RE = re.compile(r"\b(limited|ltd|private|pvt|llp|inc|corp|corporation|co)\b")
PUNCT_RE = re.compile(r"[^\w\s]")
SPACE_RE = re.compile(r"\s+")


def normalize(name: str) -> str:
    s = name.lower().strip()
    s = PUNCT_RE.sub(" ", s)
    s = SUFFIX_RE.sub(" ", s)
    s = SPACE_RE.sub(" ", s).strip()
    return s


def main():
    # 1. Combine all financial export files
    files = sorted(glob.glob(f"{DATA_DIR}/*.xlsx"))
    fin = pd.concat([pd.read_excel(f, header=7) for f in files], ignore_index=True)
    fin_companies = fin["Company"].dropna().unique().tolist()

    # 2. Load IPO company list
    ipo = pd.read_excel(IPO_FILE, sheet_name="IPO_FINAL_CLEANED_FOR_PLOTTING")
    ipo_companies = ipo["COMPANY NAME"].dropna().unique().tolist()

    # 3. Build normalized lookup for financial companies
    fin_norm_map = {}
    for c in fin_companies:
        fin_norm_map.setdefault(normalize(c), c)
    fin_norm_list = list(fin_norm_map.keys())

    rows = []
    for ipo_name in ipo_companies:
        norm_name = normalize(ipo_name)

        if norm_name in fin_norm_map:
            rows.append({
                "IPO Company Name": ipo_name,
                "Match Type": "exact_normalized",
                "Matched Financial Company": fin_norm_map[norm_name],
                "Match Score": 100.0,
                "Alt Match 2": "",
                "Alt Score 2": None,
                "Confirm (y/n)": "y",
            })
            continue

        # Fuzzy match: get top 2 candidates
        candidates = process.extract(
            norm_name, fin_norm_list, scorer=fuzz.token_sort_ratio, limit=2
        )
        best = candidates[0] if len(candidates) > 0 else None
        second = candidates[1] if len(candidates) > 1 else None

        rows.append({
            "IPO Company Name": ipo_name,
            "Match Type": "fuzzy" if best else "no_match",
            "Matched Financial Company": fin_norm_map[best[0]] if best else "",
            "Match Score": round(best[1], 1) if best else None,
            "Alt Match 2": fin_norm_map[second[0]] if second else "",
            "Alt Score 2": round(second[1], 1) if second else None,
            "Confirm (y/n)": "",
        })

    review = pd.DataFrame(rows)

    # Sort so ambiguous/fuzzy rows needing review float to the top
    review["_sort"] = review["Match Type"].map(
        {"exact_normalized": 1, "fuzzy": 0, "no_match": 0}
    )
    review = review.sort_values(["_sort", "Match Score"], ascending=[True, True]).drop(columns="_sort")

    review.to_excel(OUTPUT_FILE, index=False)

    n_exact = (review["Match Type"] == "exact_normalized").sum()
    n_fuzzy = (review["Match Type"] == "fuzzy").sum()
    n_none = (review["Match Type"] == "no_match").sum()

    print(f"Total IPO companies: {len(ipo_companies)}")
    print(f"Exact normalized matches: {n_exact}")
    print(f"Fuzzy matches (need review): {n_fuzzy}")
    print(f"No match found: {n_none}")
    print(f"\nWrote {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
