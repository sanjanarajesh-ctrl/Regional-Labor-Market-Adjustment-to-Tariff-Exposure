# Project structure and run instructions

This note explains how the repository is organized on disk and how the Python scripts
expect files to be arranged. It is meant to be read alongside `about_this_work.md`,
which describes the research itself.

## 1. Directory layout

All project files are stored directly in the main project directory
`TradeShocksPipeline/`. The analysis was originally developed in PyCharm with file
paths that reference the project root, so the scripts assume a flat structure with no
subfolders.

A typical layout looks like:

- `TradeShocksPipeline/`
  - Python scripts:
    - `trade_shocks_pipeline.py`
    - `trade_shocks_analysis.py`
    - `trade_shocks_analysis_robustness.py`
    - `figure_plotter.py`
  - Raw and processed data:
    - `2015_annual_by_area.zip`, `2016_annual_by_area.zip`, ..., `2019_annual_by_area.zip`
    - `acs_county_controls_2019.csv`
    - `bds2023_st_cty.csv`
    - `bds_county_2015_2019.csv`
    - `bea_county_gdp_2015_2019.csv`
    - `CAGDP2.zip`
    - `qcew_county_industry_2015_2019.csv`
    - `ustr_tariff_list_combined.csv`
    - `trade_shock_panel_county_2015_2019.csv`
    - `tariff_exposure_by_county.csv`
    - `tariff_naics_mapping.csv`
  - Results and figures:
    - `did_log_emp_results.txt`
    - `did_log_emp_weighted_results.txt`
    - `did_log_gdp_results.txt`
    - `did_log_gdp_weighted_results.txt`
    - `did_log_emp_heterogeneity_income.txt`
    - `did_log_emp_heterogeneity_ba_plus.txt`
    - `resilience_delta_log_emp_results.txt`
    - `event_study_log_emp_coeffs.csv`
    - `event_study_log_gdp_coeffs.csv`
    - `figure1_event_study_log_emp.png`
    - `figure2_event_study_log_gdp.png`
  - Documentation:
    - `about_this_work.md`
    - `PROJECT_STRUCTURE.md` (this file)

To run the code exactly as written, keep all of these files together in the same
directory and do not move them into subfolders.

## 2. Running the code

The scripts use relative paths that assume the current working directory is the
project root (`TradeShocksPipeline/`). A typical workflow is:

1. Open a terminal in `TradeShocksPipeline/`.
2. Activate a Python environment with the following packages installed:
   - `pandas`
   - `numpy`
   - `statsmodels`
   - `matplotlib`
3. Run the scripts as needed, for example:

   ```bash
   python trade_shocks_pipeline.py
   python trade_shocks_analysis.py
   python trade_shocks_analysis_robustness.py
   python figure_plotter.py

   
Small path edits may be required depending on the local setup, but no changes to the logic of the scripts are necessary.

## 3. Description of result files

This section explains what each main result file represents.

### 3.1 Baseline difference in differences

- **`did_log_emp_results.txt`**  
  Unweighted baseline regression of log county employment on the tariff exposure  
  index interacted with a post dummy (2018–2019), with county and year fixed effects.

- **`did_log_emp_weighted_results.txt`**  
  Same specification as above, but weighted by county employment so that larger  
  labor markets receive more weight.

- **`did_log_gdp_results.txt`**  
  Baseline difference in differences regression for log county GDP (unweighted).

- **`did_log_gdp_weighted_results.txt`**  
  Weighted version of the GDP specification.

These four files correspond to the baseline results table in the paper (Table 1).

### 3.2 Heterogeneity in employment effects

- **`did_log_emp_heterogeneity_income.txt`**  
  Regression that allows the tariff exposure effect to differ between high income  
  and low income counties (interaction of exposure × post with a high income dummy).  
  Corresponds to the income heterogeneity panel in Table 2.

- **`did_log_emp_heterogeneity_ba_plus.txt`**  
  Regression that allows the effect to differ between high BA share and low BA share  
  counties. Corresponds to the education heterogeneity panel in Table 2.

### 3.3 Resilience regression

- **`resilience_delta_log_emp_results.txt`**  
  Cross sectional regression of the change in log employment between 2017 and 2019  
  on tariff exposure and its interactions with high income, high BA share, and high  
  mobility. This is the resilience regression reported as Table 3.

### 3.4 Event study coefficients

- **`event_study_log_emp_coeffs.csv`**  
  Coefficients, standard errors, and p values for the exposure × event time  
  interactions in the employment event study. Used to construct Figure 1.

- **`event_study_log_gdp_coeffs.csv`**  
  The analogous coefficients for log county GDP. Used for Figure 2.

### 3.5 Figures

- **`figure1_event_study_log_emp.png`**  
  Event study plot of the employment response to tariff exposure, with coefficients  
  and 95 percent confidence intervals.

- **`figure2_event_study_log_gdp.png`**  
  Event study plot of the GDP response to tariff exposure.

These figures match the event study discussion in the paper.

## 4. Notes for readers

This repository is primarily intended to document the structure of the project and  
the main empirical outputs. Anyone wishing to rerun the full pipeline will need to  
have access to the underlying public data sources (USTR, BLS QCEW, BEA, Census BDS,  
ACS) and may need to adjust file paths to match their local setup. The flat directory  
layout described above reflects the original development environment and is preserved  
for transparency.

