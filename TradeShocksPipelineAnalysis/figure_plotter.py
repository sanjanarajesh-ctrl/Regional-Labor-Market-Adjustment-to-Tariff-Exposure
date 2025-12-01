import pandas as pd
import matplotlib.pyplot as plt

# Load event-study coefficient files
emp_es = pd.read_csv("event_study_log_emp_coeffs.csv")
gdp_es = pd.read_csv("event_study_log_gdp_coeffs.csv")

def ensure_ci(df):
    """
    Make sure the DataFrame has 95 percent confidence interval columns.
    If lower_95/upper_95 are missing, compute them as coef ± 1.96*se.
    """
    if "lower_95" not in df.columns or "upper_95" not in df.columns:
        df["lower_95"] = df["coef"] - 1.96 * df["se"]
        df["upper_95"] = df["coef"] + 1.96 * df["se"]
    return df

emp_es = ensure_ci(emp_es)
gdp_es = ensure_ci(gdp_es)

def plot_event_study(df, title, ylabel, outfile):
    fig, ax = plt.subplots()

    # Sort by event_time just in case
    df = df.sort_values("event_time")

    # Main line with markers and 95 percent confidence intervals as error bars
    ax.errorbar(
        df["event_time"],
        df["coef"],
        yerr=1.96 * df["se"],
        fmt="o-",
        capsize=4,
    )

    # Reference lines at 0 in time and outcome
    ax.axhline(0.0, linestyle="--", linewidth=1)
    ax.axvline(0.0, linestyle=":", linewidth=1)

    ax.set_xlabel("Event time (years relative to 2017)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # Minimal horizontal padding
    ax.set_xlim(df["event_time"].min() - 0.5, df["event_time"].max() + 0.5)

    fig.tight_layout()
    fig.savefig(outfile, dpi=300)

# Figure 1: Employment event study
plot_event_study(
    emp_es,
    title="Figure 1. Employment response to tariff exposure (event study)",
    ylabel="Exposure × year coefficient (log employment)",
    outfile="figure1_event_study_log_emp.png",
)

# Figure 2: GDP event study
plot_event_study(
    gdp_es,
    title="Figure 2. GDP response to tariff exposure (event study)",
    ylabel="Exposure × year coefficient (log GDP)",
    outfile="figure2_event_study_log_gdp.png",
)
