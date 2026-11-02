#!/usr/bin/env python3
"""Turn the raw Lending Club export into the borrower table the model trains on.

Source: accepted_2007_to_2018Q4.csv (CC0), e.g. huggingface.co/datasets/codesignal/lending-club-loan-accepted.
Keeps 36-month loans issued 2012-2015. Every one of them had reached its end date by the end of the data
(Dec 2018), so each loan has a final outcome and none are dropped for still being open.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = [
    "id", "loan_amnt", "term", "emp_length", "home_ownership", "annual_inc", "verification_status",
    "issue_d", "loan_status", "purpose", "dti", "delinq_2yrs", "earliest_cr_line", "fico_range_low",
    "fico_range_high", "inq_last_6mths", "open_acc", "pub_rec", "revol_util",
]
PURPOSES = {"debt_consolidation", "credit_card", "home_improvement", "major_purchase", "medical", "small_business", "car"}
VERIFICATION = {"Verified": "verified", "Source Verified": "source_verified", "Not Verified": "not_verified"}


def emp_years(value: str) -> float:
    if not isinstance(value, str) or value == "n/a":
        return 0.0
    if value.startswith("<"):
        return 0.5
    return float(value.split()[0].rstrip("+"))


def prepare(raw: Path) -> pd.DataFrame:
    df = pd.read_csv(raw, usecols=COLUMNS, low_memory=False)
    df = df[df["term"].str.strip() == "36 months"]
    df["issue_d"] = pd.to_datetime(df["issue_d"], format="%b-%Y")
    df = df[df["issue_d"].dt.year.between(2012, 2015)]

    status = df["loan_status"].str.replace("Does not meet the credit policy. Status:", "", regex=False)
    df = df[status.isin(["Fully Paid", "Charged Off", "Default"])]
    status = status.loc[df.index]

    earliest = pd.to_datetime(df["earliest_cr_line"], format="%b-%Y", errors="coerce")
    dti = (df["dti"].clip(0, 150) / 100).fillna(df["dti"].median() / 100)
    out = pd.DataFrame(
        {
            "application_id": "LC-" + df["id"].astype(str),
            "borrower_name": "Applicant " + df["id"].astype(str).str[-5:],
            "issue_year": df["issue_d"].dt.year,
            "annual_income": df["annual_inc"].astype(float),
            "loan_amount": df["loan_amnt"].astype(float),
            "debt_to_income": dti.round(3),
            "credit_score": ((df["fico_range_low"] + df["fico_range_high"]) / 2).round().astype(int),
            "employment_length_years": df["emp_length"].map(emp_years),
            "credit_history_length_years": ((df["issue_d"] - earliest).dt.days / 365.25).round(1),
            "revolving_utilization": (df["revol_util"] / 100).clip(0, 2).round(3),
            "num_open_accounts": df["open_acc"],
            "num_delinquencies_24m": df["delinq_2yrs"],
            "num_inquiries_6m": df["inq_last_6mths"],
            "public_records": df["pub_rec"],
            # Existing monthly debt payments implied by DTI (before the new loan).
            "monthly_debt": (df["annual_inc"] / 12 * dti).round(0),
            "home_ownership": df["home_ownership"].where(df["home_ownership"].isin(["MORTGAGE", "OWN", "RENT"]), "OTHER"),
            "loan_purpose": df["purpose"].where(df["purpose"].isin(PURPOSES), "other"),
            "verification_status": df["verification_status"].map(VERIFICATION),
            "defaulted": status.isin(["Charged Off", "Default"]).astype(int),
        }
    )
    out = out[out["annual_income"] > 0].dropna()
    for column in ("num_open_accounts", "num_delinquencies_24m", "num_inquiries_6m", "public_records"):
        out[column] = out[column].astype(int)
    return out.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/lending_club_36m_2012_2015.csv.gz"))
    args = parser.parse_args()
    frame = prepare(args.raw)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    by_year = frame.groupby("issue_year")["defaulted"].agg(["size", "mean"]).round(3)
    print(f"{len(frame):,} loans -> {args.output}\n{by_year}")


if __name__ == "__main__":
    main()
