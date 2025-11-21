import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Flow is identified by this 5-tuple
FLOW_COLS = ["srcip", "dstip", "srcport", "dstport", "proto"]


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
def _sample_array(arr: np.ndarray, max_points: int = 200_000) -> np.ndarray:
    """Randomly subsample an array if it is very large."""
    arr = np.asarray(arr)
    n = arr.size
    if n <= max_points:
        return arr
    idx = np.random.choice(n, max_points, replace=False)
    return arr[idx]


def _plot_cdf(real_vals: np.ndarray,
              syn_vals: np.ndarray,
              xlabel: str,
              title: str,
              log_x: bool = False) -> None:
    """Plot empirical CDFs of real vs synthetic arrays."""
    real_vals = np.asarray(real_vals)
    syn_vals = np.asarray(syn_vals)

    real_vals = np.sort(real_vals)
    syn_vals = np.sort(syn_vals)

    if real_vals.size == 0 or syn_vals.size == 0:
        print("[WARN] One of the datasets has zero points for this query; skipping plot.")
        return

    real_y = np.linspace(0, 1, real_vals.size, endpoint=False)
    syn_y = np.linspace(0, 1, syn_vals.size, endpoint=False)

    plt.figure(figsize=(8, 5))
    plt.plot(real_vals, real_y, label="Real", linestyle="-")
    plt.plot(syn_vals, syn_y, label="Synthetic", linestyle="--")

    if log_x:
        plt.xscale("log")

    plt.xlabel(xlabel)
    plt.ylabel("CDF")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def _flow_group(df: pd.DataFrame) -> pd.core.groupby.DataFrameGroupBy:
    """Convenience wrapper for grouping by 5-tuple flow."""
    return df.groupby(FLOW_COLS, sort=False)


# ----------------------------------------------------------------------
# Query 1: Top 10 source ports by number of packets
# ----------------------------------------------------------------------
def plot_top_srcports(real_df: pd.DataFrame,
                      syn_df: pd.DataFrame,
                      top_n: int = 10) -> None:
    """
    Compute the packet count per source port, select the top `top_n` source
    ports by count in the *real* data, and compare counts with synthetic data
    via a side-by-side bar plot.
    """
    real_counts = real_df["srcport"].value_counts()
    syn_counts = syn_df["srcport"].value_counts()

    # Take top_n ports from real data
    top_ports = real_counts.head(top_n).index.tolist()

    real_vals = real_counts.reindex(top_ports).fillna(0).astype(int)
    syn_vals = syn_counts.reindex(top_ports).fillna(0).astype(int)

    ports = [int(p) for p in top_ports]
    x = np.arange(len(ports))
    width = 0.35

    plt.figure(figsize=(10, 5))
    plt.bar(x - width / 2, real_vals.values, width=width, label="Real")
    plt.bar(x + width / 2, syn_vals.values, width=width, label="Synthetic")

    plt.xticks(x, ports, rotation=45, ha="right")
    plt.ylabel("Number of packets")
    plt.xlabel("Source port")
    plt.title(f"Top {top_n} Source Ports by Packet Count (Real vs Synthetic)")
    plt.tight_layout()
    plt.legend()
    plt.show()


# ----------------------------------------------------------------------
# Query 2: Average packet interval per flow (CDF over flows)
# ----------------------------------------------------------------------
def plot_avg_interarrival_per_flow(real_df: pd.DataFrame,
                                   syn_df: pd.DataFrame) -> None:
    """
    For each flow:
      - Sort packets by time
      - Compute inter-arrival times (time diffs between consecutive packets)
      - Take the average inter-arrival time per flow (in ms)

    Then plot the CDF of the per-flow averages (real vs synthetic).
    """

    def compute_avg_intervals(df: pd.DataFrame) -> np.ndarray:
        if df.empty:
            return np.array([])

        # Sort by flow keys + time
        df_sorted = df.sort_values(FLOW_COLS + ["time"])

        # Compute time diffs within each flow
        diffs = df_sorted.groupby(FLOW_COLS, sort=False)["time"].diff()

        # Keep only valid, non-negative diffs (drop first packet in each flow)
        valid_mask = diffs.notna() & (diffs >= 0)
        if not valid_mask.any():
            return np.array([])

        df_iat = df_sorted.loc[valid_mask].copy()
        df_iat["iat_ms"] = diffs[valid_mask] / 1000.0  # microseconds -> ms

        # Average inter-arrival per flow
        avg_per_flow = (
            df_iat.groupby(FLOW_COLS, sort=False)["iat_ms"]
            .mean()
            .to_numpy()
        )
        return avg_per_flow

    real_avg = compute_avg_intervals(real_df)
    syn_avg = compute_avg_intervals(syn_df)

    real_avg = _sample_array(real_avg)
    syn_avg = _sample_array(syn_avg)

    _plot_cdf(
        real_avg,
        syn_avg,
        xlabel="Average packet interval per flow (ms)",
        title="Avg Packet Inter-arrival Time Per Flow (Real vs Synthetic)",
        log_x=True,
    )


# ----------------------------------------------------------------------
# Query 3: Flow size distribution (# packets per flow)
# ----------------------------------------------------------------------
def plot_flow_size_distribution(real_df: pd.DataFrame,
                                syn_df: pd.DataFrame) -> None:
    """
    For each flow, count number of packets, and plot CDF of flow sizes
    (in packets) for real vs synthetic.
    """
    if real_df.empty or syn_df.empty:
        print("[WARN] One of the datasets is empty; skipping flow size distribution.")
        return

    real_sizes = _flow_group(real_df).size().to_numpy()
    syn_sizes = _flow_group(syn_df).size().to_numpy()

    real_sizes = _sample_array(real_sizes)
    syn_sizes = _sample_array(syn_sizes)

    _plot_cdf(
        real_sizes,
        syn_sizes,
        xlabel="Flow size (packets)",
        title="Flow Size Distribution (Real vs Synthetic)",
        log_x=True,
    )


# ----------------------------------------------------------------------
# Query 4: Flow byte volume (sum of pkt_len per flow)
# ----------------------------------------------------------------------
def plot_flow_byte_volume_distribution(real_df: pd.DataFrame,
                                       syn_df: pd.DataFrame) -> None:
    """
    For each flow, compute total bytes = sum(pkt_len),
    and plot CDF of flow byte volumes for real vs synthetic.
    """
    if real_df.empty or syn_df.empty:
        print("[WARN] One of the datasets is empty; skipping flow byte volume distribution.")
        return

    real_bytes = _flow_group(real_df)["pkt_len"].sum().to_numpy()
    syn_bytes = _flow_group(syn_df)["pkt_len"].sum().to_numpy()

    real_bytes = _sample_array(real_bytes)
    syn_bytes = _sample_array(syn_bytes)

    _plot_cdf(
        real_bytes,
        syn_bytes,
        xlabel="Flow volume (bytes)",
        title="Flow Byte Volume Distribution (Real vs Synthetic)",
        log_x=True,
    )


# ----------------------------------------------------------------------
# Query 5: Packet size distribution (pkt_len)
# ----------------------------------------------------------------------
def plot_packet_size_distribution(real_df: pd.DataFrame,
                                  syn_df: pd.DataFrame) -> None:
    """
    Visualize packet size distribution as a bar chart using bins:
    <0, 0-100, 100-200, ..., 1400-1500, 1501+ (real vs synthetic).
    """

    import numpy as np

    if "pkt_len" not in real_df.columns or "pkt_len" not in syn_df.columns:
        print("[WARN] 'pkt_len' column not found; skipping packet size distribution.")
        return

    # Define bins and labels
    bins = [-1, 0] + list(range(100, 1600, 100)) + [np.inf]
    labels = ["<0"] + \
             [f"{i}-{i+100}" for i in range(0, 1500, 100)] + \
             ["1501+"]

    def bin_counts(df):
        return pd.cut(df["pkt_len"], bins=bins, labels=labels, right=True).value_counts().reindex(labels).fillna(0)

    real_hist = bin_counts(real_df)
    syn_hist  = bin_counts(syn_df)

    # Bar plot
    x = np.arange(len(labels))
    width = 0.4

    plt.figure(figsize=(14, 6))
    plt.bar(x - width/2, real_hist.values, width=width, label="Real")
    plt.bar(x + width/2, syn_hist.values, width=width, label="Synthetic")

    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("Packet Count")
    plt.xlabel("Packet Length Bin (bytes)")
    plt.title("Packet Size Distribution (Real vs Synthetic)")
    plt.tight_layout()
    plt.legend()
    plt.show()


# ----------------------------------------------------------------------
# Driver: run all 5 queries
# ----------------------------------------------------------------------
def eval_queries(real_df: pd.DataFrame, syn_df: pd.DataFrame) -> None:
    """
    Run 5 PCAP-style evaluation queries sequentially.

      1. Top 10 source ports by number of packets
      2. Average packet inter-arrival time per flow (CDF over flows)
      3. Flow size distribution (# packets per flow)
      4. Flow byte volume distribution (sum of pkt_len per flow)
      5. Packet size distribution (pkt_len)

    For each query:
      - Print the natural-language description
      - Produce a visualization comparing real vs synthetic.
    """
    queries = [
        (
            "Which source ports have the most packets (top 10 srcports by packet count)?",
            plot_top_srcports,
        ),
        (
            "For each flow, what is the average packet interval, and how does the CDF "
            "over flows compare (real vs synthetic)?",
            plot_avg_interarrival_per_flow,
        ),
        (
            "How many packets does each flow contain (flow size distribution)?",
            plot_flow_size_distribution,
        ),
        (
            "For each flow, what is the total byte volume (sum of pkt_len)?",
            plot_flow_byte_volume_distribution,
        ),
        (
            "What is the distribution of packet sizes (pkt_len) in real vs synthetic traffic?",
            plot_packet_size_distribution,
        ),
    ]

    for i, (desc, func) in enumerate(queries, start=1):
        print("=" * 80)
        print(f"Query {i}: {desc}")
        print("-" * 80)
        try:
            func(real_df, syn_df)
        except Exception as e:
            print(f"[Error running query {i}] {e}")
        print("\n")

