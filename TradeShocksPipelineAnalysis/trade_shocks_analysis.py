"""
Trade Policy Shocks & Regional Resilience: Estimation Script

Assumes you have already run trade_shocks_pipeline.py and created:
    trade_shock_panel_county_2015_2019.csv

This script:
  1. Loads the panel and restricts to counties.
  2. Constructs exposure, outcomes, and controls.
  3. Runs baseline DiD for multiple outcomes.
  4. Runs an event study for log employment.
  5. Runs heterogeneity regressions (which county traits dampen the effect).
  6. Builds a simple "resilience" measure (pre to post change) and links it to traits.

Outputs:
  - did_log_emp_results.txt
  - did_log_gdp_results.txt
  - did_firm_birth_rate_results.txt
  - event_study_log_emp_coeffs.csv
  - did_log_emp_heterogeneity_results.txt
  - resilience_delta_log_emp_results.txt
"""

import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
PANEL_PATH = os.path.join(DATA_DIR, "trade_shock_panel_county_2015_2019.csv")


# -------------------------------------------------------------------
# 1. Load and prepare data
# -------------------------------------------------------------------

def load_and_prepare_panel() -> pd.DataFrame:
    df = pd.read_csv(PANEL_PATH, dtype={"fips": str})

    # Keep only counties (drop state totals like 01000, 02000, etc.)
    df = df[df["fips"].str[-3:] != "000"].copy()

    # Construct exposure variable:
    # prefer tariff_exposure_weighted, fall back to tariff_exposure_share
    if "tariff_exposure_weighted" in df.columns:
        exp_main = df["tariff_exposure_weighted"]
    else:
        exp_main = pd.Series(np.nan, index=df.index)

    if "tariff_exposure_share" in df.columns:
        exp_fallback = df["tariff_exposure_share"]
    else:
        exp_fallback = pd.Series(np.nan, index=df.index)

    df["exposure"] = exp_main.fillna(exp_fallback)

    # Post dummy: treat 2018–2019 as post period, 2015–2017 as pre
    df["post"] = (df["year"] >= 2018).astype(int)

    # Outcomes
    df["log_emp"] = np.log1p(df["emp_total"])
    if "gdp_current_dollars" in df.columns:
        df["log_gdp"] = np.log1p(df["gdp_current_dollars"])
    else:
        df["log_gdp"] = np.nan

    # firm_birth_rate from BDS should already exist; keep as is
    if "firm_birth_rate" not in df.columns:
        df["firm_birth_rate"] = np.nan

    # Drop rows missing exposure or main outcome
    df = df.dropna(subset=["exposure", "log_emp"])

    return df


# -------------------------------------------------------------------
# 2. Baseline DiD helper for arbitrary outcome
# -------------------------------------------------------------------

def run_baseline_did(df: pd.DataFrame, outcome: str, spec_name: str) -> None:
    """
    Baseline DiD:
        y_it = beta * (exposure_i * post_t) + alpha_i + gamma_t + e_it

    Implemented via OLS with county and year dummies,
    cluster-robust SEs by county (fips).

    Safely skips if there is no usable data for the chosen outcome.
    """

    df = df.copy()
    if outcome not in df.columns:
        print(f"[WARN] Outcome {outcome} not in dataframe. Skipping.")
        return

    # Drop rows where outcome or exposure is missing
    df = df.dropna(subset=[outcome, "exposure"])

    # If nothing left, skip
    if df.empty:
        print(f"[WARN] No non-missing observations for {outcome}. Skipping DiD.")
        return

    # Also skip if essentially no variation (e.g. 1 year or 1 county)
    if df["fips"].nunique() < 2 or df["year"].nunique() < 2:
        print(
            f"[WARN] Not enough variation for {outcome} "
            f"(fips unique={df['fips'].nunique()}, years unique={df['year'].nunique()}). Skipping."
        )
        return

    df["treat_post"] = df["exposure"] * df["post"]
    df["y"] = df[outcome]

    formula = "y ~ treat_post + C(fips) + C(year)"

    model = smf.ols(formula, data=df)
    res = model.fit(
        cov_type="cluster",
        cov_kwds={"groups": df["fips"]}
    )

    print(f"\n=== Baseline DiD for {outcome} ===")
    print(res.summary().tables[1])

    out_path = os.path.join(DATA_DIR, f"did_{spec_name}_results.txt")
    with open(out_path, "w") as f:
        f.write(res.summary().as_text())
    print(f"Saved {spec_name} DiD results to {out_path}")


# -------------------------------------------------------------------
# 3. Event-study specification for log_emp
# -------------------------------------------------------------------

def run_event_study(df: pd.DataFrame):
    """
    Event-study around 2017 as the reference year.

    event_time = year - 2017 in {-2, -1, 0, 1, 2}

    Regression:
        log_emp_it = sum_k beta_k * (exposure_i * 1[event_time_t = k])
                     + alpha_i + gamma_t + e_it

    with k in {-2, -1, 1, 2}, leaving 0 as base.
    """

    df = df.copy()
    df = df.dropna(subset=["log_emp", "exposure"])

    df["event_time"] = df["year"] - 2017
    valid_event_times = [-2, -1, 0, 1, 2]
    df = df[df["event_time"].isin(valid_event_times)]

    mapping = {
        -2: "m2",
        -1: "m1",
        1: "p1",
        2: "p2"
    }

    event_terms = []
    for k, suffix in mapping.items():
        dummy_name = f"D_{suffix}"
        inter_name = f"exp_D_{suffix}"

        df[dummy_name] = (df["event_time"] == k).astype(int)
        df[inter_name] = df["exposure"] * df[dummy_name]
        event_terms.append(inter_name)

    rhs = " + ".join(event_terms) + " + C(fips) + C(year)"
    formula = f"log_emp ~ {rhs}"

    model = smf.ols(formula, data=df)
    res = model.fit(
        cov_type="cluster",
        cov_kwds={"groups": df["fips"]}
    )

    print("\n=== Event-study: log_emp on exposure × event-time dummies ===")
    print(res.summary().tables[1])

    rows = []
    for k, suffix in mapping.items():
        name = f"exp_D_{suffix}"
        if name in res.params.index:
            rows.append({
                "event_time": k,
                "coef": res.params[name],
                "se": res.bse[name]
            })

    es_df = pd.DataFrame(rows).sort_values("event_time")
    es_df["lower_95"] = es_df["coef"] - 1.96 * es_df["se"]
    es_df["upper_95"] = es_df["coef"] + 1.96 * es_df["se"]

    out_path = os.path.join(DATA_DIR, "event_study_log_emp_coeffs.csv")
    es_df.to_csv(out_path, index=False)
    print(f"Saved event-study coefficients to {out_path}")

    return res, es_df


# -------------------------------------------------------------------
# 4. Heterogeneity in treatment effects by county characteristics
# -------------------------------------------------------------------

def run_heterogeneity(df: pd.DataFrame):
    """
    Heterogeneity in treatment effect by county characteristics.

    We run three separate regressions:
        log_emp ~ treat_post + treat_post × Z_std + C(fips) + C(year)

    where Z is, in turn:
      - median_income
      - pct_ba_plus
      - pct_moved_diff_state

    Each spec is run separately for numerical stability. If a variable is
    missing or has ~zero variance, that spec is skipped.
    """

    df = df.copy()
    df = df.dropna(subset=["log_emp", "exposure"])

    df["treat_post"] = df["exposure"] * df["post"]

    controls = [
        ("median_income", "income"),
        ("pct_ba_plus", "ba_plus"),
        ("pct_moved_diff_state", "mobility"),
    ]

    for col, short in controls:
        if col not in df.columns:
            print(f"[WARN] Heterogeneity: {col} not in dataframe. Skipping.")
            continue

        sub = df.dropna(subset=[col, "treat_post", "log_emp"]).copy()
        if sub.empty:
            print(f"[WARN] Heterogeneity: no non-missing data for {col}. Skipping.")
            continue

        # Standardize control; skip if almost no variation
        mean = sub[col].mean()
        std = sub[col].std(ddof=0)
        if std < 1e-8 or np.isnan(std):
            print(f"[WARN] Heterogeneity: {col} has ~zero variance. Skipping.")
            continue

        z_name = f"{col}_std"
        sub[z_name] = (sub[col] - mean) / std
        sub["tp_x_z"] = sub["treat_post"] * sub[z_name]

        formula = "log_emp ~ treat_post + tp_x_z + C(fips) + C(year)"

        try:
            model = smf.ols(formula, data=sub)
            res = model.fit(
                cov_type="cluster",
                cov_kwds={"groups": sub["fips"]}
            )
        except np.linalg.LinAlgError:
            print(f"[WARN] Heterogeneity regression for {col} failed (SVD did not converge). Skipping.")
            continue

        print(f"\n=== Heterogeneity: log_emp with {col} ===")
        print(res.summary().tables[1])

        out_path = os.path.join(
            DATA_DIR, f"did_log_emp_heterogeneity_{short}.txt"
        )
        with open(out_path, "w") as f:
            f.write(res.summary().as_text())
        print(f"Saved heterogeneity results for {col} to {out_path}")


# -------------------------------------------------------------------
# 5. Resilience as pre to post change (cross sectional)
# -------------------------------------------------------------------

def build_resilience_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a county-level dataset where the outcome is
      delta_log_emp_17_19 = log_emp_2019 - log_emp_2017

    Treat 2017 as last pre year and 2019 as post.
    Exposure is the mean exposure in 2018–2019.
    Controls are taken from 2017 (income, education, mobility, firm birth rate).
    """

    df = df.copy()
    df = df.dropna(subset=["log_emp", "exposure"])

    # Exposure: average over post years
    exp_post = (
        df[df["year"] >= 2018]
        .groupby("fips")["exposure"]
        .mean()
        .rename("exp_post")
    )

    # Outcomes: pivot log_emp for 2017 and 2019
    sub = df[df["year"].isin([2017, 2019])].copy()
    wide = sub.pivot(index="fips", columns="year", values="log_emp")
    # Ensure both years present
    wide = wide.dropna(subset=[2017, 2019])

    wide = wide.rename(columns={2017: "log_emp_2017", 2019: "log_emp_2019"})
    wide["delta_log_emp_17_19"] = wide["log_emp_2019"] - wide["log_emp_2017"]

    # Controls from 2017
    controls_cols = [
        "median_income",
        "pct_ba_plus",
        "pct_moved_diff_state",
        "firm_birth_rate",
    ]
    controls_2017 = (
        df[df["year"] == 2017][["fips"] + controls_cols]
        .drop_duplicates("fips")
        .set_index("fips")
    )

    # Combine
    res_df = wide.join(exp_post, how="inner")
    res_df = res_df.join(controls_2017, how="left")

    # Drop rows with missing key variables
    res_df = res_df.dropna(subset=["delta_log_emp_17_19", "exp_post"])

    return res_df.reset_index()


def run_resilience_regression(res_df: pd.DataFrame):
    """
    Cross sectional regression:
        delta_log_emp_17_19 = beta * exp_post
                             + gamma_1 exp_post × income_std
                             + gamma_2 exp_post × edu_std
                             + gamma_3 exp_post × mobility_std
                             + controls + error
    """

    df = res_df.copy()

    controls = ["median_income", "pct_ba_plus", "pct_moved_diff_state"]
    df = df.dropna(subset=controls)

    # Standardize controls
    for c in controls:
        mean = df[c].mean()
        std = df[c].std(ddof=0)
        df[c + "_std"] = (df[c] - mean) / std

    df["tp"] = df["exp_post"]
    df["tp_x_inc"] = df["tp"] * df["median_income_std"]
    df["tp_x_ba"] = df["tp"] * df["pct_ba_plus_std"]
    df["tp_x_move"] = df["tp"] * df["pct_moved_diff_state_std"]

    formula = (
        "delta_log_emp_17_19 ~ tp + tp_x_inc + tp_x_ba + tp_x_move"
    )

    model = smf.ols(formula, data=df)
    res = model.fit(cov_type="HC1")  # robust SEs

    print("\n=== Resilience regression: change in log_emp 2017–2019 ===")
    print(res.summary().tables[1])

    out_path = os.path.join(DATA_DIR, "resilience_delta_log_emp_results.txt")
    with open(out_path, "w") as f:
        f.write(res.summary().as_text())
    print(f"Saved resilience regression results to {out_path}")

    return res


# -------------------------------------------------------------------
# 6. Main
# -------------------------------------------------------------------

def main():
    df = load_and_prepare_panel()

    print("Panel after county restriction and cleaning:")
    print(df.groupby("year")["fips"].nunique())

    # Baseline DiD for three outcomes
    run_baseline_did(df, outcome="log_emp", spec_name="log_emp")
    run_baseline_did(df, outcome="log_gdp", spec_name="log_gdp")
    run_baseline_did(df, outcome="firm_birth_rate", spec_name="firm_birth_rate")

    # Event study for log_emp
    run_event_study(df)

    # Heterogeneity in treatment effect
    run_heterogeneity(df)

    # Resilience as pre to post change
    res_df = build_resilience_dataset(df)
    print(f"\nResilience dataset shape: {res_df.shape}")
    run_resilience_regression(res_df)


if __name__ == "__main__":
    main()
