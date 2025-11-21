import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple

HIGH_INCOME_LABEL = ">50K"  # change here if your label is different

# ---------- Generic JSD helpers ----------
def _safe_normalize(x):
    x = np.asarray(x, dtype=float)
    total = x.sum()
    if total <= 0:
        return np.ones_like(x) / len(x)
    return x / total

def jsd_from_probs(p, q):
    p = _safe_normalize(p)
    q = _safe_normalize(q)
    m = 0.5 * (p + q)

    def _kl(a, b):
        mask = a > 0
        a = a[mask]
        b = b[mask]
        return np.sum(a * np.log(a / b))

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


# ---------- Column-wise JSD evaluator ----------
def eval_jsd(real_df: pd.DataFrame,
                         syn_df: pd.DataFrame,
                         num_bins: int = 100):
    """
    Compute JSD for each column's distribution (real vs synthetic).
    - For categorical columns: use value_counts().
    - For numeric columns: histogram with `num_bins` bins (shared range).
    Returns:
        {
          'col_jsd': {column_name: jsd_value, ...},
          'avg_jsd': mean_over_all_columns
        }
    """
    results = {}
    col_jsd = {}

    for col in real_df.columns:
        if col not in syn_df.columns:
            continue  # skip missing columns

        # --- Categorical case ---
        if real_df[col].dtype == object or str(real_df[col].dtype).startswith("category"):
            real_counts = real_df[col].value_counts()
            syn_counts = syn_df[col].value_counts()

            # Align categories
            all_keys = sorted(set(real_counts.index) | set(syn_counts.index))
            real_vec = real_counts.reindex(all_keys).fillna(0).to_numpy()
            syn_vec  = syn_counts.reindex(all_keys).fillna(0).to_numpy()

            jsd = jsd_from_probs(real_vec, syn_vec)
            col_jsd[col] = float(jsd)
            continue

        # --- Numeric case ---
        try:
            real_vals = real_df[col].dropna().to_numpy(dtype=float)
            syn_vals  = syn_df[col].dropna().to_numpy(dtype=float)
        except:
            # fallback to categorical handling
            real_counts = real_df[col].astype(str).value_counts()
            syn_counts  = syn_df[col].astype(str).value_counts()
            all_keys = sorted(set(real_counts.index) | set(syn_counts.index))
            real_vec = real_counts.reindex(all_keys).fillna(0).to_numpy()
            syn_vec  = syn_counts.reindex(all_keys).fillna(0).to_numpy()
            jsd = jsd_from_probs(real_vec, syn_vec)
            col_jsd[col] = float(jsd)
            continue

        # If numeric but constant => 0 divergence if both constant
        if real_vals.size == 0 or syn_vals.size == 0:
            col_jsd[col] = 1.0
            continue

        # Build shared numeric bins
        lo = min(real_vals.min(), syn_vals.min())
        hi = max(real_vals.max(), syn_vals.max())
        bins = np.linspace(lo, hi, num_bins + 1)

        real_hist, _ = np.histogram(real_vals, bins=bins)
        syn_hist, _  = np.histogram(syn_vals,  bins=bins)

        jsd = jsd_from_probs(real_hist, syn_hist)
        col_jsd[col] = float(jsd)

    # average over all columns
    avg_jsd = float(np.mean(list(col_jsd.values())))

    results["col_jsd"] = col_jsd
    results["avg_jsd"] = avg_jsd
    return results

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _rate_by_group(
    df: pd.DataFrame,
    group_col: str,
    income_col: str = "income",
    positive: str = HIGH_INCOME_LABEL,
) -> pd.DataFrame:
    """
    Compute the fraction of rows with income == positive for each value in group_col.
    Returns a DataFrame with columns [group_col, 'rate'] where rate is in [0, 1].
    """
    grp = (
        df.groupby(group_col)[income_col]
        .apply(lambda s: (s == positive).mean())
        .reset_index(name="rate")
    )
    return grp


def _align_real_syn(
    real_df: pd.DataFrame,
    syn_df: pd.DataFrame,
    group_col: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute >50K rates for real & synthetic data and align on common groups.

    Returns:
        simple_df: columns ['group', 'rate_real', 'rate_syn']
        full_df:   same as simple_df (kept for compatibility / extension)
    """
    real_rates = _rate_by_group(real_df, group_col).rename(
        columns={group_col: "group", "rate": "rate_real"}
    )
    syn_rates = _rate_by_group(syn_df, group_col).rename(
        columns={group_col: "group", "rate": "rate_syn"}
    )

    merged = pd.merge(real_rates, syn_rates, on="group", how="inner")
    simple_df = merged[["group", "rate_real", "rate_syn"]]
    return simple_df, merged


# ----------------------------------------------------------------------
# Query 1: Occupations with highest percentage of >50K earners
# ----------------------------------------------------------------------
def plot_income_rate_by_occupation(
    real_df: pd.DataFrame,
    syn_df: pd.DataFrame,
    top_n: int = 10,
) -> None:
    """
    Compare real vs synthetic percentage of >50K income by occupation.
    Shows top_n occupations by real >50K percentage.
    """
    merged_simple, _ = _align_real_syn(real_df, syn_df, "occupation")

    # convert to percentages
    merged_simple["rate_real_pct"] = merged_simple["rate_real"] * 100
    merged_simple["rate_syn_pct"] = merged_simple["rate_syn"] * 100

    top = merged_simple.sort_values("rate_real_pct", ascending=False).head(top_n)

    x = range(len(top))
    width = 0.35

    plt.figure(figsize=(10, 5))
    plt.bar([i - width / 2 for i in x], top["rate_real_pct"], width=width, label="Real")
    plt.bar([i + width / 2 for i in x], top["rate_syn_pct"], width=width, label="Synthetic")

    plt.xticks(x, top["group"], rotation=45, ha="right")
    plt.ylabel("Percentage with income >50K (%)")
    plt.title(f"Top {top_n} Occupations by >50K Rate (Real vs Synthetic)")
    plt.ylim(0, 100)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------------------
# Query 2 (replacing race): Marital-status vs >50K percentage
# ----------------------------------------------------------------------
def plot_income_rate_by_marital_status(
    real_df: pd.DataFrame,
    syn_df: pd.DataFrame,
) -> None:
    """
    Compare real vs synthetic percentage of >50K income by marital-status.
    """
    merged_simple, _ = _align_real_syn(real_df, syn_df, "marital-status")

    merged_simple["rate_real_pct"] = merged_simple["rate_real"] * 100
    merged_simple["rate_syn_pct"] = merged_simple["rate_syn"] * 100

    x = range(len(merged_simple))
    width = 0.35

    plt.figure(figsize=(10, 5))
    plt.bar(
        [i - width / 2 for i in x], merged_simple["rate_real_pct"], width=width, label="Real"
    )
    plt.bar(
        [i + width / 2 for i in x], merged_simple["rate_syn_pct"], width=width, label="Synthetic"
    )

    plt.xticks(x, merged_simple["group"], rotation=45, ha="right")
    plt.ylabel("Percentage with income >50K (%)")
    plt.title("Income >50K Rate by Marital Status (Real vs Synthetic)")
    plt.ylim(0, 100)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------------------
# Query 3: Education vs >50K percentage
# ----------------------------------------------------------------------
def plot_income_rate_by_education(
    real_df: pd.DataFrame,
    syn_df: pd.DataFrame,
    education_col: str = "education",
) -> None:
    """
    Compare real vs synthetic percentage of >50K income by education level.
    """
    merged_simple, _ = _align_real_syn(real_df, syn_df, education_col)

    merged_simple["rate_real_pct"] = merged_simple["rate_real"] * 100
    merged_simple["rate_syn_pct"] = merged_simple["rate_syn"] * 100

    merged_simple = merged_simple.sort_values("rate_real_pct", ascending=False)

    x = range(len(merged_simple))
    width = 0.35

    plt.figure(figsize=(10, 5))
    plt.bar(
        [i - width / 2 for i in x], merged_simple["rate_real_pct"], width=width, label="Real"
    )
    plt.bar(
        [i + width / 2 for i in x], merged_simple["rate_syn_pct"], width=width, label="Synthetic"
    )

    plt.xticks(x, merged_simple["group"], rotation=45, ha="right")
    plt.ylabel("Percentage with income >50K (%)")
    plt.title("Income >50K Rate by Education Level (Real vs Synthetic)")
    plt.ylim(0, 100)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------------------
# Query 4 (replacing country): Average hours-per-week by income class
# ----------------------------------------------------------------------
def plot_avg_hours_by_income(
    real_df: pd.DataFrame,
    syn_df: pd.DataFrame,
    hours_col: str = "hours-per-week",
    income_col: str = "income",
) -> None:
    """
    Compare real vs synthetic average hours-per-week for each income class
    (<=50K vs >50K, or whatever labels are present in the income column).
    """
    real_mean = (
        real_df.groupby(income_col)[hours_col].mean().reset_index(name="mean_real")
    )
    syn_mean = syn_df.groupby(income_col)[hours_col].mean().reset_index(name="mean_syn")

    merged = pd.merge(real_mean, syn_mean, on=income_col, how="inner")
    merged = merged.sort_values(income_col)

    x = range(len(merged))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar([i - width / 2 for i in x], merged["mean_real"], width=width, label="Real")
    plt.bar([i + width / 2 for i in x], merged["mean_syn"], width=width, label="Synthetic")

    plt.xticks(x, merged[income_col])
    plt.ylabel("Average hours per week")
    plt.title("Average Hours per Week by Income Class (Real vs Synthetic)")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------------------
# Query 5: Age bracket vs >50K percentage
# ----------------------------------------------------------------------
def plot_income_rate_by_age_bracket(
    real_df: pd.DataFrame,
    syn_df: pd.DataFrame,
    bin_width: int = 10,
    age_col: str = "age",
) -> None:
    """
    Compare real vs synthetic percentage of >50K income by age bracket.
    Age brackets are defined as [k*bin_width, (k+1)*bin_width - 1].
    """

    def add_age_bin(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        min_age = df[age_col].min()
        max_age = df[age_col].max()
        # choose bin edges covering the full range
        bins = list(
            range(
                (min_age // bin_width) * bin_width,
                ((max_age // bin_width) + 2) * bin_width,
                bin_width,
            )
        )
        labels = [f"{bins[i]}–{bins[i + 1] - 1}" for i in range(len(bins) - 1)]
        df["age_bin"] = pd.cut(
            df[age_col], bins=bins, labels=labels, right=True, include_lowest=True
        )
        return df

    real_binned = add_age_bin(real_df)
    syn_binned = add_age_bin(syn_df)

    merged_simple, _ = _align_real_syn(real_binned, syn_binned, "age_bin")

    merged_simple["rate_real_pct"] = merged_simple["rate_real"] * 100
    merged_simple["rate_syn_pct"] = merged_simple["rate_syn"] * 100

    merged_simple["age_bin"] = pd.Categorical(
        merged_simple["group"], categories=merged_simple["group"], ordered=True
    )
    merged_simple = merged_simple.sort_values("age_bin")

    x = range(len(merged_simple))
    width = 0.35

    plt.figure(figsize=(10, 5))
    plt.bar(
        [i - width / 2 for i in x], merged_simple["rate_real_pct"], width=width, label="Real"
    )
    plt.bar(
        [i + width / 2 for i in x], merged_simple["rate_syn_pct"], width=width, label="Synthetic"
    )

    plt.xticks(x, merged_simple["group"], rotation=45, ha="right")
    plt.ylabel("Percentage with income >50K (%)")
    plt.title(
        f"Income >50K Rate by Age Bracket (Real vs Synthetic, bin={bin_width} years)"
    )
    plt.ylim(0, 100)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------------------
# Driver: run all 5 queries
# ----------------------------------------------------------------------
def eval_query(real_df: pd.DataFrame, syn_df: pd.DataFrame) -> None:
    """
    Run 5 evaluation queries sequentially:
      1. Occupation vs % >50K
      2. Marital-status vs % >50K
      3. Education vs % >50K
      4. Avg hours-per-week by income class
      5. Age brackets vs % >50K

    For each query:
      - Print the natural-language description
      - Produce the comparison visualization
    """

    queries = [
        (
            "Which occupations have the highest percentage of >50K earners?",
            plot_income_rate_by_occupation,
        ),
        (
            "For each marital-status, what percentage of individuals earn >50K?",
            plot_income_rate_by_marital_status,
        ),
        (
            "How does the percentage of >50K earners vary across education levels?",
            plot_income_rate_by_education,
        ),
        (
            "For each income class (<=50K vs >50K), what is the average hours-per-week?",
            plot_avg_hours_by_income,
        ),
        (
            "Which 10-year age brackets have the highest percentage of >50K earners?",
            plot_income_rate_by_age_bracket,
        ),
    ]

    for i, (description, func) in enumerate(queries, start=1):
        print("=" * 80)
        print(f"Query {i}: {description}")
        print("-" * 80)
        try:
            func(real_df, syn_df)
        except Exception as e:
            print(f"[Error running query {i}] {e}")
        print("\n")
