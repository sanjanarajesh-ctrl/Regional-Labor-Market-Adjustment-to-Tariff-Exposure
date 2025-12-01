"""
Data Construction Pipeline

This script builds a county-year panel for 2015–2019 combining:
- BLS QCEW (employment, wages by county-industry)
- USTR → NAICS tariffs (tariff_naics_mapping.csv)
- BDS county (firm births, deaths, firms, employment)
- BEA CAGDP2 (county GDP, current dollars)
- ACS 5-year (population, income, education, mobility) via Census API

Outputs:
- trade_shock_panel_county_2015_2019.csv

"""

import os
import zipfile
import re

import pandas as pd
import requests


# =============================================================================
# CONFIG
# =============================================================================

# Directory where all your input files live
DATA_DIR = os.path.dirname(os.path.abspath(__file__))

YEARS_QCEW = [2015, 2016, 2017, 2018, 2019]

QCEW_ZIPS = {
    year: os.path.join(DATA_DIR, f"{year}_annual_by_area.zip")
    for year in YEARS_QCEW
}

TARIFF_NAICS_PATH = os.path.join(DATA_DIR, "tariff_naics_mapping.csv")
BDS_PATH = os.path.join(DATA_DIR, "bds2023_st_cty.csv")
BEA_ZIP_PATH = os.path.join(DATA_DIR, "CAGDP2.zip")

# Optional Census API key (not required)
CENSUS_API_KEY = os.environ.get("CENSUS_API_KEY", None)

CENSUS_BASE_URL = "https://api.census.gov/data"


# =============================================================================
# ACS HELPERS
# =============================================================================

def fetch_acs_table(year: int,
                    dataset: str,
                    variables: dict,
                    geo: str = "county:*") -> pd.DataFrame:
    """
    Generic helper for pulling ACS tables via the Census API.

    Parameters
    ----------
    year : int
        ACS year to query (e.g., 2019).
    dataset : str
        Dataset path segment, e.g. "acs/acs5" or "acs/acs5/subject".
    variables : dict
        Mapping from 'nice_name' -> 'ACS variable code', e.g.
        {"population": "B01003_001E"}.
    geo : str
        Geographic filter. "county:*" means all counties.

    Returns
    -------
    DataFrame with columns: fips, state, county, NAME, and the named variables.
    """
    base_url = f"{CENSUS_BASE_URL}/{year}/{dataset}"
    var_codes = list(variables.values())
    get_clause = ",".join(["NAME"] + var_codes)

    params = {
        "get": get_clause,
        "for": geo
    }
    if CENSUS_API_KEY:
        params["key"] = CENSUS_API_KEY

    resp = requests.get(base_url, params=params)
    resp.raise_for_status()
    data = resp.json()

    # First row is header
    header = data[0]
    rows = data[1:]
    df = pd.DataFrame(rows, columns=header)

    # County FIPS
    if "state" in df.columns and "county" in df.columns:
        df["fips"] = df["state"].astype(str).str.zfill(2) + df["county"].astype(str).str.zfill(3)
    else:
        df["fips"] = None

    # Rename ACS variable codes to friendly names
    inv_var_map = {v: k for k, v in variables.items()}
    rename_cols = {code: inv_var_map[code] for code in var_codes}
    df = df.rename(columns=rename_cols)

    keep_cols = ["fips", "state", "county", "NAME"] + list(variables.keys())
    return df[keep_cols]


def build_acs_controls(year: int = 2019) -> pd.DataFrame:
    """
    Fetches county-level ACS controls for a given year and returns a single
    merged DataFrame with:

      - population
      - median_income
      - pct_ba_plus
      - pct_moved_diff_state

    using ACS 5-year estimates.
    """

    # 1) Population
    pop_vars = {"population": "B01003_001E"}
    df_pop = fetch_acs_table(year, "acs/acs5", pop_vars)

    # 2) Median income
    inc_vars = {"median_income": "B19013_001E"}
    df_inc = fetch_acs_table(year, "acs/acs5", inc_vars)

    # 3) Education (percent BA+)
    edu_vars = {"pct_ba_plus": "S1501_C02_015E"}
    df_edu = fetch_acs_table(year, "acs/acs5/subject", edu_vars)

    # 4) Mobility (percent who moved from a different state)
    mob_vars = {"pct_moved_diff_state": "S0701_C04_001E"}
    df_mob = fetch_acs_table(year, "acs/acs5/subject", mob_vars)

    # Merge on fips
    df = df_pop.merge(df_inc, on=["fips", "state", "county", "NAME"], how="left")
    df = df.merge(df_edu, on=["fips", "state", "county", "NAME"], how="left")
    df = df.merge(df_mob, on=["fips", "state", "county", "NAME"], how="left")

    # Numeric conversions
    for col in ["population", "median_income", "pct_ba_plus", "pct_moved_diff_state"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    acs = df.rename(columns={"NAME": "county_name"})
    return acs


# =============================================================================
# QCEW LOAD AND PANEL BUILD
# =============================================================================

def load_qcew_year(zip_path: str, year: int) -> pd.DataFrame:
    """
    Load a single QCEW annual_by_area ZIP file and return area–industry data.

    Memory-safe version:
      - reads ALL CSV/TXT files in the ZIP
      - only loads the columns we actually need
      - filters per-file on numeric area_fips
      - constructs a 5-digit FIPS from area_fips
    """
    import zipfile
    import pandas as pd
    import os

    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"QCEW zip not found: {zip_path}")

    usecols = [
        "area_fips",
        "industry_code",
        "year",
        "annual_avg_emplvl",
        "annual_avg_wkly_wage",
    ]

    dfs = []

    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            if not name.lower().endswith((".csv", ".txt")):
                continue

            with z.open(name) as f:
                try:
                    df_part = pd.read_csv(
                        f,
                        usecols=usecols,
                        dtype={"area_fips": str, "industry_code": str},
                        low_memory=False,
                    )
                except ValueError:
                    # In case some files have slightly different columns,
                    # read full then subset.
                    f.seek(0)
                    df_full = pd.read_csv(
                        f,
                        dtype={"area_fips": str, "industry_code": str},
                        low_memory=False,
                    )
                    # add any missing columns as NA
                    for c in usecols:
                        if c not in df_full.columns:
                            df_full[c] = pd.NA
                    df_part = df_full[usecols].copy()

            # Ensure year exists
            if "year" not in df_part.columns:
                df_part["year"] = year
            df_part["year"] = pd.to_numeric(df_part["year"], errors="coerce")

            # Filter on numeric area_fips per file, to avoid creating a huge mask later
            df_part["area_fips"] = df_part["area_fips"].astype(str).str.strip()
            mask_numeric = df_part["area_fips"].str.isnumeric()
            df_part = df_part.loc[mask_numeric].copy()

            # Build 5-digit FIPS
            df_part["fips"] = df_part["area_fips"].str.zfill(5).str[:5]

            # Numeric conversions
            df_part["annual_avg_emplvl"] = pd.to_numeric(
                df_part["annual_avg_emplvl"], errors="coerce"
            )
            df_part["annual_avg_wkly_wage"] = pd.to_numeric(
                df_part["annual_avg_wkly_wage"], errors="coerce"
            )

            dfs.append(
                df_part[["fips", "industry_code", "year", "annual_avg_emplvl", "annual_avg_wkly_wage"]]
            )

    if not dfs:
        raise ValueError(f"No CSV/TXT files found inside {zip_path}")

    df = pd.concat(dfs, ignore_index=True)

    # Debug output
    print(f"QCEW {year}: total rows after numeric area_fips filter = {len(df)}")
    print(f"QCEW {year}: unique fips count = {df['fips'].nunique()}")

    return df



def build_qcew_panel() -> pd.DataFrame:
    """
    Build combined QCEW panel for 2015–2019, county × industry.
    """
    frames = []
    for y in YEARS_QCEW:
        print(f"Loading QCEW for {y}...")
        df_y = load_qcew_year(QCEW_ZIPS[y], y)
        frames.append(df_y)

    qcew = pd.concat(frames, ignore_index=True)
    qcew.to_csv(os.path.join(DATA_DIR, "qcew_county_industry_2015_2019.csv"), index=False)
    print("Saved qcew_county_industry_2015_2019.csv")
    return qcew


# =============================================================================
# TARIFF → NAICS AND EXPOSURE
# =============================================================================

def load_tariff_naics_mapping() -> pd.DataFrame:
    """
    Load tariff_naics_mapping.csv and collapse to NAICS6-level tariff indicators.
    """
    df = pd.read_csv(TARIFF_NAICS_PATH, dtype=str)
    df["naics6"] = df["naics6"].str.strip()

    # Clean tariff_rate to numeric percent if possible
    rate_num = (
        df["tariff_rate"].astype(str)
        .str.replace("%", "", regex=False)
        .str.strip()
    )
    df["tariff_rate_num"] = pd.to_numeric(rate_num, errors="coerce") / 100.0

    # Collapse to NAICS6
    grouped = df.groupby("naics6", as_index=False).agg(
        hs_count=("hs8", "nunique"),
        avg_tariff=("tariff_rate_num", "mean")
    )
    grouped["tariff_dummy"] = (grouped["hs_count"] > 0).astype(int)

    return grouped


def compute_tariff_exposure(qcew: pd.DataFrame,
                            tariff_naics: pd.DataFrame) -> pd.DataFrame:
    """
    Compute county-level tariff exposure using 2017 baseline employment shares.

    Steps:
      - Filter QCEW to year 2017.
      - Map industry_code to 6-digit NAICS.
      - Merge with tariff_naics.
      - Construct employment shares by county-industry.
      - Exposure = sum over industries of share * tariff_dummy (and weighted version).
    """
    q17 = qcew[qcew["year"] == 2017].copy()
    q17["naics6"] = q17["industry_code"].str[:6]

    base = q17.merge(tariff_naics, on="naics6", how="left")

    base["tariff_dummy"] = base["tariff_dummy"].fillna(0)
    base["avg_tariff"] = base["avg_tariff"].fillna(0.0)

    # Total employment by county (base year)
    emp_tot = base.groupby("fips", as_index=False)["annual_avg_emplvl"].sum()
    emp_tot = emp_tot.rename(columns={"annual_avg_emplvl": "emp_total_2017"})
    base = base.merge(emp_tot, on="fips", how="left")

    base["emp_share"] = base["annual_avg_emplvl"] / base["emp_total_2017"]
    base["emp_share"] = base["emp_share"].fillna(0)

    base["share_tariff_dummy"] = base["emp_share"] * base["tariff_dummy"]
    base["share_tariff_weighted"] = base["emp_share"] * base["avg_tariff"]

    exposure = base.groupby("fips", as_index=False).agg(
        tariff_exposure_share=("share_tariff_dummy", "sum"),
        tariff_exposure_weighted=("share_tariff_weighted", "sum"),
        emp_total_2017=("emp_total_2017", "first")
    )

    exposure.to_csv(os.path.join(DATA_DIR, "tariff_exposure_by_county.csv"), index=False)
    print("Saved tariff_exposure_by_county.csv")

    return exposure


# =============================================================================
# BDS: COUNTY BIRTHS / DEATHS
# =============================================================================
def build_bds_county() -> pd.DataFrame:
    """
    Parse BDS county file (bds2023_st_cty.csv) and extract 2015–2019.

    This version is robust to different column names for state/county,
    e.g. it will work whether the file uses:
      - fipsstate / fipscounty
      - st / cty
      - state / county
      - or a single 'fips' column
    """
    bds = pd.read_csv(BDS_PATH, dtype=str)

    # --- Identify state and county code columns, or an existing fips column ---
    state_col = None
    county_col = None

    # If there's already a fips column, just use it
    if "fips" in bds.columns:
        pass  # we'll use it later
    else:
        # Try common patterns for state code
        for cand in ["fipsstate", "st", "state", "STATE"]:
            if cand in bds.columns:
                state_col = cand
                break

        # Try common patterns for county code
        for cand in ["fipscounty", "cty", "county", "COUNTY"]:
            if cand in bds.columns:
                county_col = cand
                break

        if state_col is None or county_col is None:
            raise KeyError(
                f"Could not find state/county columns in BDS file. "
                f"Available columns: {list(bds.columns)}"
            )

        # Build a 5-digit county FIPS from state + county
        bds[state_col] = bds[state_col].str.zfill(2)
        bds[county_col] = bds[county_col].str.zfill(3)
        bds["fips"] = bds[state_col] + bds[county_col]

    # --- Year filter ---
    if "year" not in bds.columns:
        raise KeyError("BDS file does not have a 'year' column.")

    bds["year"] = pd.to_numeric(bds["year"], errors="coerce")
    bds = bds[(bds["year"] >= 2015) & (bds["year"] <= 2019)].copy()

    # --- Identify key numeric columns (firms, emp, births, deaths) ---
    possible_cols = {
        "firms": ["firms", "FIRMS"],
        "estabs": ["estabs", "ESTABS"],
        "emp": ["emp", "EMP"],
        "births": ["firm_births", "births", "BIRTHS"],
        "deaths": ["firm_deaths", "deaths", "DEATHS"],
    }

    def find_col(df, options):
        for col in options:
            if col in df.columns:
                return col
        return None

    col_firms = find_col(bds, possible_cols["firms"])
    col_emp = find_col(bds, possible_cols["emp"])
    col_births = find_col(bds, possible_cols["births"])
    col_deaths = find_col(bds, possible_cols["deaths"])

    # Convert to numeric where present
    for col in [col_firms, col_emp, col_births, col_deaths]:
        if col and col in bds.columns:
            bds[col] = pd.to_numeric(bds[col], errors="coerce")

    # Compute firm birth rate if possible
    if col_births and col_firms:
        bds["firm_birth_rate"] = bds[col_births] / bds[col_firms]
    else:
        bds["firm_birth_rate"] = None

    # Build output subset
    keep_cols = ["fips", "year"]
    if col_firms:
        keep_cols.append(col_firms)
    if col_emp:
        keep_cols.append(col_emp)
    if col_births:
        keep_cols.append(col_births)
    if col_deaths:
        keep_cols.append(col_deaths)
    keep_cols.append("firm_birth_rate")

    bds_out = bds[keep_cols].copy()

    # Rename to stable names
    rename_map = {}
    if col_firms:
        rename_map[col_firms] = "firms"
    if col_emp:
        rename_map[col_emp] = "bds_emp"
    if col_births:
        rename_map[col_births] = "firm_births"
    if col_deaths:
        rename_map[col_deaths] = "firm_deaths"

    bds_out = bds_out.rename(columns=rename_map)

    out_path = os.path.join(DATA_DIR, "bds_county_2015_2019.csv")
    bds_out.to_csv(out_path, index=False)
    print("Saved bds_county_2015_2019.csv")

    return bds_out




# =============================================================================
# BEA: COUNTY GDP
# =============================================================================

def build_bea_county_gdp() -> pd.DataFrame:
    """
    Read CAGDP2.zip and extract county-level GDP (current dollars),
    2015–2019, line 1 (all industry total).
    """
    if not os.path.exists(BEA_ZIP_PATH):
        raise FileNotFoundError(f"BEA CAGDP2 zip not found: {BEA_ZIP_PATH}")

    with zipfile.ZipFile(BEA_ZIP_PATH) as z:
        csv_name = None
        for name in z.namelist():
            if name.lower().endswith(".csv"):
                csv_name = name
                break
        if csv_name is None:
            raise ValueError("No CSV found inside CAGDP2.zip")

        with z.open(csv_name) as f:
            #gdp = pd.read_csv(f, dtype={"GeoFIPS": str})
            raw = f.read()
            text = raw.decode("latin-1", errors="replace")
        import io
        gdp = pd.read_csv(io.StringIO(text), dtype={"GeoFIPS": str})
    # Clean FIPS
    gdp["GeoFIPS"] = gdp["GeoFIPS"].astype(str)
    gdp["fips"] = gdp["GeoFIPS"].str.extract(r"(\d+)")[0].str.zfill(5)

    # LineCode 1: All industry total
    gdp = gdp[gdp["LineCode"] == 1].copy()

    # Wide to long
    year_cols = [str(y) for y in range(2015, 2020) if str(y) in gdp.columns]
    gdp_long = gdp.melt(
        id_vars=["fips"],
        value_vars=year_cols,
        var_name="year",
        value_name="gdp_current_dollars"
    )
    gdp_long["year"] = gdp_long["year"].astype(int)
    gdp_long["gdp_current_dollars"] = pd.to_numeric(
        gdp_long["gdp_current_dollars"],
        errors="coerce"
    )

    gdp_long.to_csv(os.path.join(DATA_DIR, "bea_county_gdp_2015_2019.csv"), index=False)
    print("Saved bea_county_gdp_2015_2019.csv")

    return gdp_long


# =============================================================================
# FINAL PANEL BUILD
# =============================================================================

def build_final_panel():
    # 1) QCEW panel
    qcew = build_qcew_panel()

    # 2) Collapse QCEW to county-year totals (for main outcomes)
    county_panel = qcew.groupby(["fips", "year"], as_index=False).agg(
        emp_total=("annual_avg_emplvl", "sum"),
        avg_weekly_wage=("annual_avg_wkly_wage", "mean")
    )

    # 3) Tariff exposure from 2017 baseline
    tariff_naics = load_tariff_naics_mapping()
    exposure = compute_tariff_exposure(qcew, tariff_naics)

    # 4) BDS county
    bds = build_bds_county()

    # 5) BEA GDP
    bea_gdp = build_bea_county_gdp()

    # 6) ACS controls (2019 5-year)
    acs = build_acs_controls(year=2019)
    acs.to_csv(os.path.join(DATA_DIR, "acs_county_controls_2019.csv"), index=False)
    print("Saved acs_county_controls_2019.csv")

    # Merge everything
    panel = county_panel.merge(exposure, on="fips", how="left")
    panel = panel.merge(bds, on=["fips", "year"], how="left")
    panel = panel.merge(bea_gdp, on=["fips", "year"], how="left")

    # ACS is time-invariant here, so merge only by fips
    panel = panel.merge(acs, on="fips", how="left")

    out_path = os.path.join(DATA_DIR, "trade_shock_panel_county_2015_2019.csv")
    panel.to_csv(out_path, index=False)
    print("Saved trade_shock_panel_county_2015_2019.csv with shape:", panel.shape)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    build_final_panel()
