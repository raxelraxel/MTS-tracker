"""
fetch_mts_data.py

Fetches the latest monthly Treasury MTS data (Tables 2, 3, 4, 5) and BEA GDP
deflator, cleans and combines them into a single row, then appends that row to
data/mts_data.csv if the month isn't already present.
"""

import requests
import pandas as pd
from datetime import date
from pathlib import Path
import sys

# ── Config ────────────────────────────────────────────────────────────────────
CSV_PATH = Path("data/mts_data.csv")
DATE_COL = "record_date"

BEA_USER_ID = "4EF114EC-EABF-4B44-A0C4-C2598C8CBCAF"  # move to env var / secret if desired


# ── API URLs ──────────────────────────────────────────────────────────────────
def build_urls(record_date: str) -> dict:
    """
    Build all API URLs using a date string like '2026-04-30'.
    Tables 3, 4, 5 filter by exact record_date; Table 2 returns multiple months
    and we filter in Python.
    """
    return {
        "table2": (
            "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v1/accounting/mts/mts_table_2?"
            "fields=record_date,record_fiscal_year,record_calendar_month,current_month_budget_amt,"
            "current_fytd_budget_amt,classification_desc,line_code_nbr"
            "&filter=line_code_nbr:in:(20,50,110,130,140,150)"
            f",record_date:eq:{record_date}"
            "&sort=-record_date&format=json"
        ),
        "table3": (
            "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v1/accounting/mts/mts_table_3?"
            "fields=record_date,record_fiscal_year,record_calendar_month,current_month_rcpt_outly_amt,"
            "current_fytd_rcpt_outly_amt,classification_desc,line_code_nbr"
            f"&filter=record_date:eq:{record_date}"
            "&sort=-record_date&format=json"
        ),
        "table4": (
            "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v1/accounting/mts/mts_table_4?"
            "fields=record_date,record_fiscal_year,record_calendar_month,current_month_net_rcpt_amt,"
            "current_fytd_net_rcpt_amt,classification_desc,line_code_nbr"
            f"&filter=line_code_nbr:in:(102),record_date:eq:{record_date}"
            "&sort=-record_date&format=json"
        ),
        "table5": (
            "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v1/accounting/mts/mts_table_5?"
            "fields=record_date,record_fiscal_year,record_calendar_month,current_month_net_outly_amt,"
            "current_fytd_net_outly_amt,classification_desc,line_code_nbr"
            f"&filter=line_code_nbr:in:(1480,3091,4150,4152,4148,2788,2790,2797,2848,4525,4560)"
            f",record_date:eq:{record_date}"
            "&sort=-record_date&format=json"
        ),
    }


BEA_URL = "https://apps.bea.gov/api/data/"
BEA_PARAMS = {
    "UserID": BEA_USER_ID,
    "datasetname": "NIPA",
    "TableName": "T10109",
    "ResultFormat": "JSON",
    "method": "GETDATA",
    "Frequency": "Q",
    "Year": "2016,2017,2018,2019,2020,2021,2022,2023,2024,2025,2026",
}

# SeriesCode for the GDP deflator in BEA T10109
BEA_DEFLATOR_SERIES = "DPCERD"


# ── Helpers ───────────────────────────────────────────────────────────────────
def fetch_json(url: str, params: dict = None) -> dict:
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def get_val(data: list[dict], line_code: str, field: str) -> float | str:
    """Find a row by line_code_nbr and return its numeric field value."""
    row = next((r for r in data if str(r.get("line_code_nbr")) == str(line_code)), None)
    if row is None:
        return ""
    v = row.get(field)
    if v is None or v == "" or v == "null":
        return ""
    try:
        return float(v)
    except (ValueError, TypeError):
        return ""


def get_str(data: list[dict], field: str) -> str:
    """Return a string field from the first row."""
    if not data:
        return ""
    return str(data[0].get(field, ""))


def to_num(v: float | str) -> float:
    return 0.0 if v == "" else float(v)


def date_to_bea_quarter(record_date: str) -> str:
    """
    Convert a record_date string (YYYY-MM-DD) to BEA quarter format (e.g. '2026Q1').
    The OfficeScript handled Excel serial numbers; the API returns ISO dates directly.
    """
    dt = pd.to_datetime(record_date)
    quarter = f"Q{dt.quarter}"
    return f"{dt.year}{quarter}"


# ── BEA deflator ──────────────────────────────────────────────────────────────
def fetch_deflator_rows() -> list[dict]:
    print("  Fetching BEA GDP deflator...")
    payload = fetch_json(BEA_URL, BEA_PARAMS)
    try:
        rows = payload["BEAAPI"]["Results"]["Data"]
    except (KeyError, TypeError):
        print("  WARNING: Could not parse BEA deflator data.")
        return []
    # Filter to the deflator series only
    return [r for r in rows if r.get("SeriesCode") == BEA_DEFLATOR_SERIES]


def get_deflator_value(record_date: str, deflator_rows: list[dict]) -> float | str:
    """Exact quarter match, falling back to the most recent available quarter."""
    if not record_date or not deflator_rows:
        return ""
    target = date_to_bea_quarter(record_date)
    # Exact match
    match = next((r for r in deflator_rows if r.get("TimePeriod") == target), None)
    if match:
        return float(match["DataValue"].replace(",", ""))
    # Fallback: most recent quarter
    sorted_rows = sorted(deflator_rows, key=lambda r: r.get("TimePeriod", ""))
    return float(sorted_rows[-1]["DataValue"].replace(",", ""))


# ── Main fetch & clean ─────────────────────────────────────────────────────────
def fetch_and_clean(record_date: str) -> pd.DataFrame:
    """
    Call all 5 APIs, clean, and return a single-row DataFrame
    matching the column layout from the OfficeScript.
    """
    urls = build_urls(record_date)

    print(f"  Fetching Table 2...")
    t2 = fetch_json(urls["table2"]).get("data", [])
    print(f"  Fetching Table 3...")
    t3 = fetch_json(urls["table3"]).get("data", [])
    print(f"  Fetching Table 4...")
    t4 = fetch_json(urls["table4"]).get("data", [])
    print(f"  Fetching Table 5...")
    t5 = fetch_json(urls["table5"]).get("data", [])
    deflator_rows = fetch_deflator_rows()

    if not t2:
        print(f"  WARNING: Table 2 returned no data for {record_date}. Skipping.")
        return pd.DataFrame()

    # ── Shared fields ──────────────────────────────────────────────────────────
    rd          = get_str(t2, "record_date")
    fiscal_year = get_str(t2, "record_fiscal_year")
    cal_month   = get_str(t2, "record_calendar_month")

    # ── Table 2 ────────────────────────────────────────────────────────────────
    current_gross_receipts          = get_val(t2, "20",  "current_month_budget_amt")
    current_gross_outlays           = get_val(t2, "50",  "current_month_budget_amt")
    current_month_deficit           = get_val(t2, "110", "current_month_budget_amt")
    financing_borrowing             = get_val(t2, "130", "current_month_budget_amt")
    financing_reduce_operating_cash = get_val(t2, "140", "current_month_budget_amt")
    financing_other                 = get_val(t2, "150", "current_month_budget_amt")

    # ── Table 3 ────────────────────────────────────────────────────────────────
    individual_income_tax           = get_val(t3, "20",  "current_month_rcpt_outly_amt")
    corp_income_tax                 = get_val(t3, "30",  "current_month_rcpt_outly_amt")
    emp_retire_off                  = get_val(t3, "50",  "current_month_rcpt_outly_amt")
    emp_retire_on                   = get_val(t3, "60",  "current_month_rcpt_outly_amt")
    ui                              = get_val(t3, "70",  "current_month_rcpt_outly_amt")
    other_retire                    = get_val(t3, "80",  "current_month_rcpt_outly_amt")
    excise                          = get_val(t3, "90",  "current_month_rcpt_outly_amt")
    estate_gift                     = get_val(t3, "100", "current_month_rcpt_outly_amt")
    customs                         = get_val(t3, "110", "current_month_rcpt_outly_amt")
    misc                            = get_val(t3, "120", "current_month_rcpt_outly_amt")
    total_receipts                  = get_val(t3, "130", "current_month_rcpt_outly_amt")
    total_receipts_on               = get_val(t3, "140", "current_month_rcpt_outly_amt")
    total_receipts_off              = get_val(t3, "150", "current_month_rcpt_outly_amt")
    leg                             = get_val(t3, "170", "current_month_rcpt_outly_amt")
    jud                             = get_val(t3, "180", "current_month_rcpt_outly_amt")
    usda                            = get_val(t3, "210", "current_month_rcpt_outly_amt")
    doc                             = get_val(t3, "220", "current_month_rcpt_outly_amt")
    dod_mil                         = get_val(t3, "230", "current_month_rcpt_outly_amt")
    ed                              = get_val(t3, "250", "current_month_rcpt_outly_amt")
    doe                             = get_val(t3, "260", "current_month_rcpt_outly_amt")
    hhs                             = get_val(t3, "275", "current_month_rcpt_outly_amt")
    dhs                             = get_val(t3, "280", "current_month_rcpt_outly_amt")
    hud                             = get_val(t3, "290", "current_month_rcpt_outly_amt")
    doi                             = get_val(t3, "300", "current_month_rcpt_outly_amt")
    doj                             = get_val(t3, "310", "current_month_rcpt_outly_amt")
    dol                             = get_val(t3, "320", "current_month_rcpt_outly_amt")
    state                           = get_val(t3, "330", "current_month_rcpt_outly_amt")
    dot                             = get_val(t3, "340", "current_month_rcpt_outly_amt")
    debt_interest_gross             = get_val(t3, "360", "current_month_rcpt_outly_amt")
    treasury_other                  = get_val(t3, "370", "current_month_rcpt_outly_amt")
    va                              = get_val(t3, "375", "current_month_rcpt_outly_amt")
    corps_engineers                 = get_val(t3, "377", "current_month_rcpt_outly_amt")
    civil_defense                   = get_val(t3, "378", "current_month_rcpt_outly_amt")
    epa                             = get_val(t3, "380", "current_month_rcpt_outly_amt")
    eop                             = get_val(t3, "385", "current_month_rcpt_outly_amt")
    gsa                             = get_val(t3, "390", "current_month_rcpt_outly_amt")
    intl                            = get_val(t3, "395", "current_month_rcpt_outly_amt")
    nasa                            = get_val(t3, "400", "current_month_rcpt_outly_amt")
    nsf                             = get_val(t3, "405", "current_month_rcpt_outly_amt")
    opm                             = get_val(t3, "410", "current_month_rcpt_outly_amt")
    sba                             = get_val(t3, "420", "current_month_rcpt_outly_amt")
    ssa                             = get_val(t3, "423", "current_month_rcpt_outly_amt")
    ind_agencies                    = get_val(t3, "440", "current_month_rcpt_outly_amt")
    undistributed_offsets_interest  = get_val(t3, "470", "current_month_rcpt_outly_amt")
    undistributed_offsets_other     = get_val(t3, "480", "current_month_rcpt_outly_amt")
    total_outlays                   = get_val(t3, "490", "current_month_rcpt_outly_amt")
    total_outlays_on                = get_val(t3, "500", "current_month_rcpt_outly_amt")
    total_outlays_off               = get_val(t3, "510", "current_month_rcpt_outly_amt")
    deficit                         = get_val(t3, "520", "current_month_rcpt_outly_amt")
    deficit_on                      = get_val(t3, "530", "current_month_rcpt_outly_amt")
    deficit_off                     = get_val(t3, "540", "current_month_rcpt_outly_amt")

    # ── Table 4 / 5 ────────────────────────────────────────────────────────────
    receipts_income_tax_withheld    = get_val(t4, "102",  "current_month_net_rcpt_amt")
    outlays_oasi                    = get_val(t5, "4525", "current_month_net_outly_amt")
    outlays_di                      = get_val(t5, "4560", "current_month_net_outly_amt")
    outlays_medicaid                = get_val(t5, "2788", "current_month_net_outly_amt")
    outlays_chip                    = get_val(t5, "2790", "current_month_net_outly_amt")
    outlays_hosp_insurance_trust    = get_val(t5, "2797", "current_month_net_outly_amt")
    outlays_supp_med_trust          = get_val(t5, "2848", "current_month_net_outly_amt")
    outlays_fema                    = get_val(t5, "3091", "current_month_net_outly_amt")
    outlays_ptc                     = get_val(t5, "4148", "current_month_net_outly_amt")
    outlays_eitc                    = get_val(t5, "4150", "current_month_net_outly_amt")
    outlays_ctc                     = get_val(t5, "4152", "current_month_net_outly_amt")
    outlays_snap                    = get_val(t5, "1480", "current_month_net_outly_amt")

    # ── Derived columns ────────────────────────────────────────────────────────
    outlays_medicaidchip        = to_num(outlays_medicaid) + to_num(outlays_chip)
    outlays_refundable_credits  = to_num(outlays_ptc) + to_num(outlays_eitc) + to_num(outlays_ctc)
    gdp_deflator                = get_deflator_value(rd, deflator_rows)

    # ── Assemble row (column order matches OfficeScript newRow.push order) ─────
    row = {
        "record_date":                      rd,
        "record_fiscal_year":               fiscal_year,
        "record_calendar_month":            cal_month,
        "gdp_deflator":                     gdp_deflator,
        "current_gross_receipts":           current_gross_receipts,
        "current_gross_outlays":            current_gross_outlays,
        "current_month_deficit":            current_month_deficit,
        "financing_borrowing":              financing_borrowing,
        "financing_reduce_operating_cash":  financing_reduce_operating_cash,
        "financing_other":                  financing_other,
        "individual_income_tax":            individual_income_tax,
        "corp_income_tax":                  corp_income_tax,
        "emp_retire_off":                   emp_retire_off,
        "emp_retire_on":                    emp_retire_on,
        "ui":                               ui,
        "other_retire":                     other_retire,
        "excise":                           excise,
        "estate_gift":                      estate_gift,
        "customs":                          customs,
        "misc":                             misc,
        "total_receipts":                   total_receipts,
        "total_receipts_on":                total_receipts_on,
        "total_receipts_off":               total_receipts_off,
        "leg":                              leg,
        "jud":                              jud,
        "usda":                             usda,
        "doc":                              doc,
        "dod_mil":                          dod_mil,
        "ed":                               ed,
        "doe":                              doe,
        "hhs":                              hhs,
        "dhs":                              dhs,
        "hud":                              hud,
        "doi":                              doi,
        "doj":                              doj,
        "dol":                              dol,
        "state":                            state,
        "dot":                              dot,
        "debt_interest_gross":              debt_interest_gross,
        "treasury_other":                   treasury_other,
        "va":                               va,
        "corps_engineers":                  corps_engineers,
        "civil_defense":                    civil_defense,
        "epa":                              epa,
        "eop":                              eop,
        "gsa":                              gsa,
        "intl":                             intl,
        "nasa":                             nasa,
        "nsf":                              nsf,
        "opm":                              opm,
        "sba":                              sba,
        "ssa":                              ssa,
        "ind_agencies":                     ind_agencies,
        "undistributed_offsets_interest":   undistributed_offsets_interest,
        "undistributed_offsets_other":      undistributed_offsets_other,
        "total_outlays":                    total_outlays,
        "total_outlays_on":                 total_outlays_on,
        "total_outlays_off":                total_outlays_off,
        "deficit":                          deficit,
        "deficit_on":                       deficit_on,
        "deficit_off":                      deficit_off,
        "receipts_income_tax_withheld":     receipts_income_tax_withheld,
        "outlays_oasi":                     outlays_oasi,
        "outlays_di":                       outlays_di,
        "outlays_medicaid":                 outlays_medicaid,
        "outlays_chip":                     outlays_chip,
        "outlays_medicaidchip":             outlays_medicaidchip,
        "outlays_hosp_insurance_trust":     outlays_hosp_insurance_trust,
        "outlays_supp_med_trust":           outlays_supp_med_trust,
        "outlays_fema":                     outlays_fema,
        "outlays_ptc":                      outlays_ptc,
        "outlays_eitc":                     outlays_eitc,
        "outlays_ctc":                      outlays_ctc,
        "outlays_refundable_credits":       outlays_refundable_credits,
        "outlays_snap":                     outlays_snap,
    }

    return pd.DataFrame([row])


# ── CSV helpers ───────────────────────────────────────────────────────────────
def load_existing() -> pd.DataFrame:
    if CSV_PATH.exists():
        df = pd.read_csv(CSV_PATH)
        print(f"  → {len(df)} existing rows loaded from {CSV_PATH}")
        return df
    print(f"  → No existing CSV at {CSV_PATH}; will create a new one.")
    return pd.DataFrame()


def get_latest_record_date(existing: pd.DataFrame) -> str:
    """
    Determine the record_date to fetch.
    - If CSV is empty, fall back to last month's end date.
    - Otherwise, compute the month-end date after the most recent row.
    """
    if existing.empty:
        today = date.today()
        # Default to last month's last day
        first_of_this_month = today.replace(day=1)
        last_month_end = first_of_this_month - pd.Timedelta(days=1)
        return str(last_month_end)

    latest = pd.to_datetime(existing[DATE_COL]).max()
    # Advance one month
    next_month = latest + pd.DateOffset(months=1)
    # Get the last day of that month
    next_month_end = next_month + pd.offsets.MonthEnd(0)
    return next_month_end.strftime("%Y-%m-%d")


def append_new_row(existing: pd.DataFrame, new_row: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    """Append new_row only if its record_date isn't already in the CSV."""
    if existing.empty:
        return new_row, True

    new_date = str(new_row[DATE_COL].iloc[0])
    known_dates = set(pd.to_datetime(existing[DATE_COL]).dt.strftime("%Y-%m-%d"))
    new_date_fmt = pd.to_datetime(new_date).strftime("%Y-%m-%d")

    if new_date_fmt in known_dates:
        print(f"  → {new_date_fmt} already exists in CSV. Nothing to append.")
        return existing, False

    combined = pd.concat([existing, new_row], ignore_index=True)
    return combined, True


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    print("=== MTS Data Updater ===\n")

    existing        = load_existing()
    record_date     = get_latest_record_date(existing)
    #record_date = "2026-04-30" Hardcoded for testing purposes
    print(f"\nFetching data for record_date: {record_date}\n")
    new_row = fetch_and_clean(record_date)

    if new_row.empty:
        print("No data returned. Exiting.")
        sys.exit(0)

    combined, appended = append_new_row(existing, new_row)

    if not appended:
        sys.exit(0)

    # Sort descending by date before saving
    combined[DATE_COL] = pd.to_datetime(combined[DATE_COL])
    combined = combined.sort_values(DATE_COL, ascending=False).reset_index(drop=True)
    combined[DATE_COL] = combined[DATE_COL].dt.strftime("%Y-%m-%d")

    # ── Re-apply BEA deflator to ALL rows (captures BEA revisions) ────────────
    print("\nRefreshing GDP deflator for all rows...")
    deflator_rows = fetch_deflator_rows()
    combined["gdp_deflator"] = combined[DATE_COL].apply(
        lambda d: get_deflator_value(d, deflator_rows)
    )

    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(CSV_PATH, index=False)
    print(f"\n✓ Appended 1 new row for {record_date}. CSV now has {len(combined)} total rows.")
    print(f"  Saved to {CSV_PATH}")


if __name__ == "__main__":
    main()