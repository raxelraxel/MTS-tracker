
import pandas as pd
import numpy as np
import os
from pathlib import Path
from groq import Groq
from dotenv import load_dotenv
load_dotenv()
import json
import statsmodels.api as sm

#os.chdir(r"C:\Users\backu\Dropbox\Data Drivers\2.0\BPC\Projects\Fiscal policy\MTS App\v1.0")
BASE_DIR = Path(__file__).parent.parent  # assumes script is in scripts/
data_path = BASE_DIR / "data/mts_data.csv"
dict_path = BASE_DIR / "data/MTS dictionary.csv"

# ── Config ────────────────────────────────────────────────────────────────────
data_path = Path("data/mts_data.csv")
dict_path = Path("data/MTS dictionary.csv")

#Load output file
output_data=pd.read_csv("data/output_data.csv")
flags_data=pd.read_csv("data/output_anomaly_flags.csv")

# ── Load data ──────────────────────────────────────────────────────────────────
def load_data():
    df = pd.read_csv(data_path)
    df=df.dropna(how='all')
    df["record_date"] = pd.to_datetime(df["record_date"])
    df = df.sort_values("record_date").reset_index(drop=True)
    return df


def load_dictionary():
    dict_df = pd.read_csv(dict_path)
    # Exclude metadata variables
    dict_df = dict_df[dict_df["group"] != "metadata"].reset_index(drop=True)
    return dict_df

EXCLUDE_COLS = [
    "record_date", "record_type", "record_fiscal_year",
    "record_calendar_month", "gdp_deflator"
]

def build_zscore_df(df_raw):
    """Z-score standardize all numeric metric columns across the full raw history."""
    df = df_raw.reset_index()  # bring record_date back as a column
    df_out = df.copy()
    df_out["record_type"] = "zscore"

    numeric_cols = [c for c in df.select_dtypes(include="number").columns
                    if c not in EXCLUDE_COLS]

    for col in numeric_cols:
        mean = df[col].mean()
        sd = df[col].std()
        df_out[col] = (df[col] - mean) / sd if sd > 0 else 0.0

    return df_out

def build_deseasonalized_df(df_raw):
    """Remove monthly seasonality from all numeric metric columns."""
    df = df_raw.reset_index()  # bring record_date back as a column
    df_out = df.copy()
    df_out["record_type"] = "deseasonalized"

    numeric_cols = [c for c in df.select_dtypes(include="number").columns
                    if c not in EXCLUDE_COLS]

    t = np.arange(len(df))
    months = df["record_date"].dt.month
    dummies = pd.get_dummies(months, prefix="m", drop_first=True,dtype=float)
    dummies.index = df.index

    for col in numeric_cols:
        series = df[col]
        if series.isna().all() or series.std() == 0:
            df_out[col] = 0.0
            continue
        X = sm.add_constant(
            pd.concat([pd.Series(t, name="t", index=df.index), dummies], axis=1)
        )
        try:
            model = sm.OLS(series, X, missing="drop").fit()
            df_out[col] = model.resid
        except Exception:
            print(f"  WARNING: deseasonalize failed for {col}: {e}")
            df_out[col] = np.nan

    return df_out

def build_gdp_deflated_df(df_raw):
    # Adjust all numeric metric columns to real dollars using the gdp_deflator
    # column already embedded in the data.
    #
    # Formula: real_value = nominal_value * (base_deflator / row_deflator)
    # Base: mean gdp_deflator across all rows with a valid deflator value.
    # gdp_deflator itself is excluded from adjustment and dropped from output
    # since it is the instrument, not a metric to be plotted.
    df = df_raw.reset_index()
    df_real = df.copy()
    df_real["record_type"] = "real"
    # Base deflator: 2026 Q1 (January, February, or March 2026)
    # Match rows where record_date falls in Q1 2026 and extract the deflator
    q1_2026_mask = (
        (df["record_date"].dt.year == 2026) &
        (df["record_date"].dt.month <= 3)
    )
    q1_2026_deflators = df.loc[q1_2026_mask, "gdp_deflator"].dropna()

    if q1_2026_deflators.empty:
        # Fallback: use most recent available deflator if 2026 Q1 not in data
        valid_deflators = df["gdp_deflator"].dropna()
        if valid_deflators.empty:
            return df  # no deflator data at all, return raw
        base_deflator = valid_deflators.iloc[-1]
    else:
        # All Q1 2026 rows should share the same deflator value -- take the first
        base_deflator = q1_2026_deflators.iloc[0]

    # Identify columns to adjust -- numeric, excluding metadata and gdp_deflator
    exclude = ["record_date","record_fiscal_year", "record_calendar_month", "gdp_deflator","record_type"]
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    cols_to_adjust = [c for c in numeric_cols if c not in exclude]

    # Apply row-wise deflation using vectorized operations
    deflator_ratio = base_deflator / df["gdp_deflator"]
    for col in cols_to_adjust:
        df_real[col] = df[col] * deflator_ratio

    return df_real


#Build dataset
df_raw = load_data()
df_raw = df_raw[df_raw["record_type"] == "raw"]  # always work from raw only
df_raw = df_raw.set_index("record_date")
df_raw.index = pd.to_datetime(df_raw.index)

# Build derived datasets
df_zscore = build_zscore_df(df_raw)
df_detrended = build_deseasonalized_df(df_raw)
df_real=build_gdp_deflated_df(df_raw)

# Rebuild mts_data.csv with all three record_type levels
df_raw_out = df_raw.reset_index()  # restore record_date as column
combined = pd.concat([df_raw_out, df_zscore, df_detrended,df_real], ignore_index=True)
combined = combined.sort_values(
    ["record_date", "record_type"]
).reset_index(drop=True)
combined["record_date"] = pd.to_datetime(combined["record_date"]).dt.strftime("%Y-%m-%d")
combined.to_csv(data_path, index=False)
print(f"✓ mts_data.csv rewritten: {len(df_raw_out)} raw rows × 4 record types = {len(combined)} total rows.")

#Z-score thresholds. For now, do 2 and 3. Can allow user to adjust later.
thresholds=[2,3]

###Function to check univariate anomalies
dict_df = load_dictionary()


def z_score(value, history):
    """Calculate z-score of value against historical series."""
    mean = history.mean()
    std = history.std()
    if std == 0:
        return 0.0
    return (value - mean) / std


def check_anomalies(df, current_date, thresholds=thresholds):
    """
    df:             DataFrame with DatetimeIndex, one column per time series
    current_date:  The new row to evaluate, as a pd.Timestamp or string (e.g. '2026-04-01')
    thresholds:      z-score cutoffs for flagging anomalies

    Returns a DataFrame with z-scores and anomaly flags for each column and each check.
    """
    current_date = pd.Timestamp(current_date)
    prev_date = current_date - pd.offsets.MonthEnd(1)

    results = {}

    for col in df.columns:
        if col in ('date_set', 'record_month','record_calendar_month','record_fiscal_year','record_type'):
            continue
        series = df[col]

        #If either the value for the column is blank in the current period, or in the previous period, skip it.
        #This may not be appropriate for all analyses.
        current_val = series[current_date]
        if pd.isna(current_val):
            continue
        prev_val = series[prev_date]
        if pd.isna(prev_val):
            continue

        #This value is used in YoY and MoM changes
        month_change = current_val - prev_val
        # ── 1) Same-month comparison ──────────────────────────────────────────
        # e.g. April 2026 vs April 2025, 2024, 2023 ...

        same_month_history = series[
            (series.index.month == current_date.month) &
            (series.index.year < current_date.year)
            ]
        z1 = z_score(current_val, same_month_history)

        # ── 2) Year-over-year monthly change ─────────────────────────────────
        # e.g. (April 2026 - March 2026) vs (April 2025 - March 2025), etc.


        # Build history of same month-over-month change in prior years
        yoy_changes = []
        for year in series.index.year.unique():
            if year >= current_date.year:
                continue
            try:
                cur = series[pd.Timestamp(year=int(year), month=current_date.month, day=1) + pd.offsets.MonthEnd(0)]
                prv = series[pd.Timestamp(year=int(year), month=prev_date.month, day=1) + pd.offsets.MonthEnd(0)]
                if pd.isna(cur) or pd.isna(prv):
                    continue
                yoy_changes.append(cur - prv)
            except KeyError:
                pass  # skip years with missing index dates

        yoy_changes = pd.Series(yoy_changes)
        z2 = z_score(month_change, yoy_changes)

        # ── 3) Consecutive month-over-month change ────────────────────────────
        # e.g. (April - March 2026) vs (March - Feb 2026), (Feb - Jan 2026), ...


        # Build history of all prior consecutive month changes
        shifted = series.shift(1)
        mom_changes = (series - shifted).dropna()
        mom_history = mom_changes[mom_changes.index < current_date]
        z3 = z_score(month_change, mom_history)

        # ── Collect results ───────────────────────────────────────────────────
        results[col] = {
                "record_date": current_date,
                # Check 1
                "val_current": current_val,
                "val_same_month_avg": same_month_history.mean(),
                #"z_same_month": round(z1, 2),
                "anomaly_same_month": 3 if abs(z1) >= thresholds[1] else 2 if abs(z1) >= thresholds[0] else 1,
                # Check 2
                "yoy_change_current": month_change,
                "yoy_change_avg": yoy_changes.mean(),
                #"z_yoy_change": round(z2, 2),
                "anomaly_yoy_change": 3 if abs(z2) >= thresholds[1] else 2 if abs(z2) >= thresholds[0] else 1,
                # Check 3
                "mom_change_current": month_change,
                "mom_change_avg": mom_history.mean(),
                #"z_mom_change": round(z3, 2),
                "anomaly_mom_change": 3 if abs(z3) >= thresholds[1] else 2 if abs(z3) >= thresholds[0] else 1,
            }

    results_df = pd.DataFrame(results).T
    results_df["variable"] = results_df.index

    return results_df

#Function to save groq summary to a json file
def save_groq_summary(record_date, summary_text):
    file_path = "data/groq_summaries.json"

    with open(file_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    records = [r for r in records if r["record_date"] != str(record_date)]

    records.append({
        "record_date": str(record_date),
        "summary": summary_text
    })

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)


###########Run the analysis code code

#Identify latest record date from the updated data.
current_date = df_raw.index.max()
record_date=current_date
output=check_anomalies(df_raw, current_date, thresholds=thresholds)
output["total_score"]=output["anomaly_same_month"]+output["anomaly_yoy_change"]+output["anomaly_mom_change"] #create total risk score
#Add output to output data
output_data = pd.concat([output, output_data])
output_data = output_data.drop_duplicates(subset=["record_date", "variable"], keep="first")
output_data.to_csv("data/output_data.csv", index=False)

#Originally Just look at the 3 and 2 flagged anomalies, but now using total score of 5 or higher. 5 could be one anomaly is 3 (high), and two are 1 (low), or two are medium (2) and 1 low (1).

#flags = output[(output["anomaly_same_month"].isin([2,3])) | (output["anomaly_yoy_change"].isin([2,3]))  | (output["anomaly_mom_change"].isin([2,3]))]
flags = output[output["total_score"]>=5]
flags = flags.merge(dict_df[["variable","description"]], on="variable", how="left") #merge in complete descriptions of variables.
flags=flags.sort_values("total_score", ascending=False) #sort based on total risk score, from high to low.

#Divide values by 1,000,000 to make it easier for Groq to interpret (LLMs aren't great at large numbers).
cols=["val_current","val_same_month_avg","yoy_change_current","yoy_change_avg","mom_change_current","mom_change_avg"]
flags[cols]=flags[cols].astype(float)
flags[cols]= (flags[cols]/1000000).round(2)

flags_data = pd.concat([flags, flags_data])
flags_data = flags_data.drop_duplicates(subset=["record_date", "variable"], keep="first")
flags_data.to_csv("data/output_anomaly_flags.csv", index=False)

#relabel variables for Groq prompt:
variable_labels = {
    "val_current": "Current month value",
    "val_same_month_avg": "Average historic value for month",
    "anomaly_same_month": "Flag for whether the current month is a potential anomaly relative to the historical average",
    "yoy_change_current": "Current month year-over-year change (e.g. April 2026 vs April 2025)",
    "yoy_change_avg": "Average month year-over-year change (e.g. historical average of month_t vs month_t-1)",
    "anomaly_yoy_change": "Flag for whether the current month year-over-year difference is a potential anomaly relative to the historical average",
    "mom_change_current": "Current month-over-month change (e.g. April 2026 vs March 2026)",
    "mom_change_avg": "Average month to month change (e.g. historical average of April vs March)",
    "anomaly_mom_change": "Flag for whether the current month-over-month difference is a potential anomaly relative to the historical average",
    "variable": "Name of the variable being analyzed",
}
flags.rename(columns=variable_labels, inplace=True)

#Send to Groq for narrative summary.
groq_key = os.environ.get("GROQ_API_KEY")
client=Groq(api_key=groq_key)
table_str=flags.to_string(index=False)
prompt = f"""
You will interpret the following output table in bullet point narrative form. The column labels define the variable definitions in the output table. 
The data are results from an analysis to detect anomalies in data reported monthly by the U.S. Department of Treasury in the Monthly Treasury Statement. 
All values are in US dollars. The table includes variables that presented an anomaly on at least one metric. The metrics include:
1) Current year value vs the 10-year average of prior values for the same month. E.g. April 2026 vs average of April for 2016-2026.
2) Current month year-over-year values vs the 10-year average. E.g April 2026-April 2025, vs average of April_t-April_t-1 for 2016-2026.
3) Current month, month-over-month values vs the 10-year average. E.g. April 2026-March 2026, vs average of April-March for 2016-2026.

Anomaly risk is flagged as 3, 2, and 1, where 3 is the highest risk, 2 is medium risk, and 1 is no risk. 
A variable could have an anomaly flagged for any of the metrics - not necessarily all of them.

Describe the rows with anomalies, saying the risk level (3=high, 2=medium, 1=none), the current value (in millions $), and the 10-year historical average for the flagged metric. 
Use the text from the variable "description" to describe the results. 
That variable contains a plain-language explanation of each row. 
Use bullet points no longer than 1 sentence. 

Output:
Start your output with this text: "For the month of (record_date), potential anomalies were found for (number of rows) line items. Anomalies were found in the following items:"
Then present a bulleted list of the results, where each bullet correspond to one row in the data. Order rows from highest to lowest total anomaly score, using variable "total_score".

{table_str}
"""

response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",  # good free-tier model
    messages=[{"role": "user", "content": prompt}],
    temperature=0.2,  # low temperature for factual/analytical output
)

summary_text = response.choices[0].message.content
print(summary_text)
save_groq_summary(record_date, summary_text)

###To add: PCA steps and save analysis output to a new file called mts_output.csv, appending the row.
