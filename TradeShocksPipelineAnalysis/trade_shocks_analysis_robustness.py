"""
Trade Policy Shocks & Regional Resilience - Robustness Script

This script assumes trade_shocks_pipeline.py has already been run and that
the panel:

    trade_shock_panel_county_2015_2019.csv

exists in the same directory.

What this script does:
  1. Reloads the panel and prepares:
       - exposure
       - post dummy
       - log_emp
       - log_gdp
  2. Runs weighted DiD specifications for:
       - log_emp (weights = emp_total)
       - log_gdp (weights = emp_total)
  3. Runs an event study for log_gdp analogous to the existing log_emp one.

Outputs:
  - did_log_emp_weighted_results.txt
  - did_log_gdp_weighted_results.txt
  - event_study_log_gdp_coeffs.csv
"""

import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# --------------------------------------------------------------------
# Paths and basic loading
# --------------------------------------------------------------------

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
PANEL_PATH = os.path.join(DATA_DIR, "trade_shock_panel_county_2015_2019.csv")


def load_and_prepare_panel() -> pd.DataFrame:
    """Load the panel and reconstruct the core variables."""
    df = pd.read_csv(PANEL_PATH, dtype={"fips": str})

    # Keep only counties, drop state totals like 01000, 02000, etc
    df = df[df["fips"].str[-3:] != "000"].copy()

    # Exposure - prefer weighted if present, else fallback
    if "tariff_exposure_weighted" in df.columns:
        exp_main = df["tariff_exposure_weighted"]
    else:
        exp_main = pd.Series(np.nan, index=df.index)

    if "tariff_exposure_share" in df.columns:
        exp_fallback = df["tariff_exposure_share"]
    else:
        exp_fallback = pd.Series(np.nan, index=df.index)

    df["exposure"] = exp_main.fillna(exp_fallback)

    # Post dummy: 2018 and 2019 are post tariff period
    df["post"] = (df["year"] >= 2018).astype(int)

    # Outcomes
    df["log_emp"] = np.log1p(df["emp_total"])
    if "gdp_current_dollars" in df.columns:
        df["log_gdp"] = np.log1p(df["gdp_current_dollars"])
    else:
        df["log_gdp"] = np.nan

    # Drop rows where exposure is missing
    df = df.dropna(subset=["exposure"])

    return df


# --------------------------------------------------------------------
# Weighted DiD helper
# --------------------------------------------------------------------

def run_weighted_did(df: pd.DataFrame, outcome: str, spec_name: str,
                     weight_col: str = "emp_total") -> None:
    """
    Weighted DiD:

        y_it = beta * (exposure_i * post_t) + alpha_i + gamma_t + e_it

    Implemented as WLS with county and year dummies,
    cluster robust standard errors by county.
    """

    df = df.copy()

    if outcome not in df.columns:
        print(f"[WARN] Outcome {outcome} not found. Skipping.")
        return

    if weight_col not in df.columns:
        print(f"[WARN] Weight column {weight_col} not found. Skipping {outcome}.")
        return

    # Drop rows with missing values
    df = df.dropna(subset=[outcome, "exposure", weight_col])

    if df.empty:
        print(f"[WARN] No usable data for {outcome}. Skipping.")
        return

    # weights - use employment as proxy for county size
    df["w"] = df[weight_col].astype(float)
    # drop zero or negative weights defensively
    df = df[df["w"] > 0].copy()
    if df.empty:
        print(f"[WARN] All weights non positive for {outcome}. Skipping.")
        return

    df["treat_post"] = df["exposure"] * df["post"]
    df["y"] = df[outcome]

    # Normalize weights so the scale is not insane (optional but safer)
    df["w_norm"] = df["w"] / df["w"].mean()

    formula = "y ~ treat_post + C(fips) + C(year)"

    model = smf.wls(formula, data=df, weights=df["w_norm"])
    res = model.fit(
        cov_type="cluster",
        cov_kwds={"groups": df["fips"]}
    )

    print(f"\n=== Weighted DiD for {outcome} (weights = {weight_col}) ===")
    print(res.summary().tables[1])

    out_path = os.path.join(DATA_DIR, f"did_{spec_name}_weighted_results.txt")
    with open(out_path, "w") as f:
        f.write(res.summary().as_text())
    print(f"Saved weighted DiD results for {outcome} to {out_path}")


# --------------------------------------------------------------------
# Event study for log_gdp
# --------------------------------------------------------------------

def run_event_study_gdp(df: pd.DataFrame):
    """
    Event study around 2017 for log_gdp:

        log_gdp_it = sum_k beta_k * (exposure_i * 1[event_time_t = k])
                     + alpha_i + gamma_t + e_it

    where event_time = year - 2017, and we keep k in {-2, -1, 0, 1, 2}
    with k = 0 omitted as the baseline.

    Coefficients and standard errors are saved to event_study_log_gdp_coeffs.csv.
    """

    df = df.copy()
    df = df.dropna(subset=["log_gdp", "exposure"])

    df["event_time"] = df["year"] - 2017
    valid_event_times = [-2, -1, 0, 1, 2]
    df = df[df["event_time"].isin(valid_event_times)]

    if df.empty:
        print("[WARN] No usable data for log_gdp event study. Skipping.")
        return

    mapping = {
        -2: "m2",
        -1: "m1",
        1: "p1",
        2: "p2",
    }

    event_terms = []
    for k, suffix in mapping.items():
        dname = f"D_{suffix}"
        iname = f"exp_D_{suffix}"
        df[dname] = (df["event_time"] == k).astype(int)
        df[iname] = df["exposure"] * df[dname]
        event_terms.append(iname)

    rhs = " + ".join(event_terms) + " + C(fips) + C(year)"
    formula = f"log_gdp ~ {rhs}"

    model = smf.ols(formula, data=df)
    res = model.fit(
        cov_type="cluster",
        cov_kwds={"groups": df["fips"]}
    )

    print("\n=== Event study for log_gdp ===")
    print(res.summary().tables[1])

    rows = []
    for k, suffix in mapping.items():
        name = f"exp_D_{suffix}"
        if name in res.params.index:
            rows.append(
                {
                    "event_time": k,
                    "coef": res.params[name],
                    "se": res.bse[name],
                }
            )

    es_df = pd.DataFrame(rows).sort_values("event_time")
    es_df["lower_95"] = es_df["coef"] - 1.96 * es_df["se"]
    es_df["upper_95"] = es_df["coef"] + 1.96 * es_df["se"]

    out_path = os.path.join(DATA_DIR, "event_study_log_gdp_coeffs.csv")
    es_df.to_csv(out_path, index=False)
    print(f"Saved log_gdp event study coefficients to {out_path}")


# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------

def main():
    df = load_and_prepare_panel()

    print("Panel summary in robustness script:")
    print(df.groupby("year")["fips"].nunique())

    # Weighted DiDs
    run_weighted_did(df, outcome="log_emp", spec_name="log_emp", weight_col="emp_total")
    run_weighted_did(df, outcome="log_gdp", spec_name="log_gdp", weight_col="emp_total")

    # Event study for log_gdp
    run_event_study_gdp(df)


if __name__ == "__main__":
    main()
