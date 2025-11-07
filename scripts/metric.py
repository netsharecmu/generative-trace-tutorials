import sys
import os
import pandas as pd
import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
from tqdm import tqdm

def read_network_packets(file_path):
    """
    Reads one or multiple CSV files containing network packet information and returns a merged DataFrame.

    Parameters:
        file_path (str | os.PathLike | Iterable[str | os.PathLike]):
            A single path or an iterable of paths to CSV file(s).

    Returns:
        pd.DataFrame: A DataFrame containing the network packet data. If multiple paths are
                      provided, all valid files are read and concatenated (row-wise).
                      When no valid file is found, returns an empty DataFrame with the expected columns.
    """
    from collections.abc import Iterable

    columns = [
        "srcip", "dstip", "srcport", "dstport", "proto", "time", "pkt_len",
        "version", "ihl", "tos", "id", "flag", "off", "ttl"
    ]

    def _read_one(p):
        return pd.read_csv(p, usecols=columns)

    def _is_multi(x):
        return isinstance(x, Iterable) and not isinstance(x, (str, bytes, os.PathLike))

    # Multiple paths: read all, skip invalids, and merge
    if _is_multi(file_path):
        dfs = []
        for p in file_path:
            try:
                dfs.append(_read_one(p))
            except FileNotFoundError:
                print(f"Error: File '{p}' not found. Skipping.")
            except ValueError as e:
                print(f"Error in '{p}': {e}. Skipping.")
        if not dfs:
            # No valid files found; return empty DF with schema
            return pd.DataFrame(columns=columns)
        return pd.concat(dfs, ignore_index=True)

    # Single path
    try:
        return _read_one(file_path)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return pd.DataFrame(columns=columns)
    except ValueError as e:
        print(f"Error: {e}")
        return pd.DataFrame(columns=columns)


def jensenshannon_wrapper(real_df_1, gen_df_2, base=2):
    """
    Computes the Jensen-Shannon Divergence (JSD) between two distributions.

    Args:
        real_df_1 (np.ndarray): First distribution.
        gen_df_2 (np.ndarray): Second distribution.
        base (int): Logarithm base for JSD calculation.

    Returns:
        float: Jensen-Shannon Divergence value.
    """
    real_df_1 = np.asarray(real_df_1, dtype=np.float64)
    gen_df_2 = np.asarray(gen_df_2, dtype=np.float64)

    if len(gen_df_2) == 0:
        return 1.0

    # Clip negatives and normalize
    gen_df_2 = np.clip(gen_df_2, 0, None)
    real_df_1 = np.clip(real_df_1, 0, None)

    # Pad to equal length
    if len(gen_df_2) < len(real_df_1):
        gen_df_2 = np.pad(gen_df_2, (0, len(real_df_1) - len(gen_df_2)))
    elif len(gen_df_2) > len(real_df_1):
        real_df_1 = np.pad(real_df_1, (0, len(gen_df_2) - len(real_df_1)))

    # Normalize to probability distributions
    real_df_1 = real_df_1 / real_df_1.sum() if real_df_1.sum() > 0 else np.zeros_like(real_df_1)
    gen_df_2 = gen_df_2 / gen_df_2.sum() if gen_df_2.sum() > 0 else np.zeros_like(gen_df_2)

    if gen_df_2.sum() == 0:
        return 1.0
    
    if real_df_1.sum() == 0:
        print("Warning: real_df_1 sum is zero, returning 1.0")
        exit(1)

    return jensenshannon(real_df_1, gen_df_2, base=base)


def wasserstein_distance_wrapper(real_df_1, gen_df_2):
    if len(gen_df_2) == 0:
        return 1.0
    else:
        return wasserstein_distance(real_df_1, gen_df_2)


def packet_stateless__count(real_df_1, gen_df_2, n=10):
    """
    Computes the Absolute Relative Error (ARE) of the total number of packets between
    the real and generated packet datasets.

    Args:
        real_df_1 (pd.DataFrame): DataFrame containing real packet data.
        gen_df_2 (pd.DataFrame): DataFrame containing generated packet data.

    Returns:
        float: Absolute Relative Error (ARE), capped at 1.0.
    """
    real_count = len(real_df_1)
    gen_count = len(gen_df_2)

    if real_count == 0:
        return 1.0 if gen_count > 0 else 0.0

    error = abs(real_count - gen_count) / real_count
    return min(error, 1.0)


def packet_stateless_srcip_countdistinct(real_df_1, gen_df_2, n=10):
    """
    Computes the Absolute Relative Error (ARE) of the number of distinct source IPs
    between the real and generated datasets.
    
    Args:
        real_df_1 (pd.DataFrame): Real dataset containing network packet data.
        gen_df_2 (pd.DataFrame): Generated dataset containing network packet data.
    
    Returns:
        float: Absolute Relative Error (ARE), capped at 1.0.
    """
    real_count = real_df_1['srcip'].nunique()
    gen_count = gen_df_2['srcip'].nunique()
    
    if real_count == 0:
        return 1.0 if gen_count > 0 else 0.0
    
    error = abs(real_count - gen_count) / real_count
    return min(error, 1.0)


def packet_stateless_srcip_distribution(real_df_1, gen_df_2, n=10):
    """
    Computes the Jensen-Shannon Divergence (JSD) between the source IP packet distributions
    of the real and generated datasets. The comparison is IP-agnostic — only the sorted 
    distribution values are compared.
    
    Args:
        real_df_1 (pd.DataFrame): Real dataset containing network packet data.
        gen_df_2 (pd.DataFrame): Generated dataset containing network packet data.
    
    Returns:
        float: Jensen-Shannon Divergence (JSD) value in [0, 1].
    """
    # Count the number of packets per source IP
    real_counts = real_df_1['srcip'].value_counts().values
    gen_counts = gen_df_2['srcip'].value_counts().values
    
    # Normalize to obtain relative distributions
    real_dist = real_counts / real_counts.sum()
    gen_dist = gen_counts / gen_counts.sum()
    
    # Sort distributions in descending order (IP-agnostic)
    real_dist_sorted = np.sort(real_dist)[::-1]
    gen_dist_sorted = np.sort(gen_dist)[::-1]
    
    # Pad the shorter array with zeros
    max_len = max(len(real_dist_sorted), len(gen_dist_sorted))
    real_padded = np.pad(real_dist_sorted, (0, max_len - len(real_dist_sorted)))
    gen_padded = np.pad(gen_dist_sorted, (0, max_len - len(gen_dist_sorted)))
    
    # Compute and return JSD
    return jensenshannon_wrapper(real_padded, gen_padded, base=2)
    

def packet_stateless_dstip_countdistinct(real_df_1, gen_df_2, n=10):
    """
    Computes the Absolute Relative Error (ARE) of the number of distinct destination IPs
    between the real and generated datasets.
    
    Args:
        real_df_1 (pd.DataFrame): Real dataset containing network packet data.
        gen_df_2 (pd.DataFrame): Generated dataset containing network packet data.
    
    Returns:
        float: Absolute Relative Error (ARE), capped at 1.0.
    """
    real_count = real_df_1['dstip'].nunique()
    gen_count = gen_df_2['dstip'].nunique()

    if real_count == 0:
        return 1.0 if gen_count > 0 else 0.0

    error = abs(real_count - gen_count) / real_count
    return min(error, 1.0)


def packet_stateless_dstip_distribution(real_df_1, gen_df_2, n=10):
    """
    Computes the Jensen-Shannon Divergence (JSD) between the destination IP packet distributions
    of the real and generated datasets. The comparison is IP-agnostic — only the sorted 
    distribution values are compared.

    Args:
        real_df_1 (pd.DataFrame): Real dataset containing network packet data.
        gen_df_2 (pd.DataFrame): Generated dataset containing network packet data.

    Returns:
        float: Jensen-Shannon Divergence (JSD) value in [0, 1].
    """
    # Count the number of packets per destination IP
    real_counts = real_df_1['dstip'].value_counts().values
    gen_counts = gen_df_2['dstip'].value_counts().values

    # Normalize to obtain relative distributions
    real_dist = real_counts / real_counts.sum()
    gen_dist = gen_counts / gen_counts.sum()

    # Sort distributions in descending order (IP-agnostic)
    real_dist_sorted = np.sort(real_dist)[::-1]
    gen_dist_sorted = np.sort(gen_dist)[::-1]

    # Pad the shorter array with zeros to match lengths
    max_len = max(len(real_dist_sorted), len(gen_dist_sorted))
    real_padded = np.pad(real_dist_sorted, (0, max_len - len(real_dist_sorted)))
    gen_padded = np.pad(gen_dist_sorted, (0, max_len - len(gen_dist_sorted)))

    # Compute and return JSD
    return jensenshannon_wrapper(real_padded, gen_padded, base=2)


def packet_stateless_srcport_countdistinct(real_df_1, gen_df_2, n=10):
    """
    Computes the Absolute Relative Error (ARE) of the number of distinct source ports
    between the real and generated datasets.
    
    Args:
        real_df_1 (pd.DataFrame): Real dataset containing network packet data.
        gen_df_2 (pd.DataFrame): Generated dataset containing network packet data.
    
    Returns:
        float: Absolute Relative Error (ARE), capped at 1.0.
    """
    real_count = real_df_1['srcport'].nunique()
    gen_count = gen_df_2['srcport'].nunique()
    
    if real_count == 0:
        return 1.0 if gen_count > 0 else 0.0
    
    error = abs(real_count - gen_count) / real_count
    return min(error, 1.0)


def packet_stateless_srcport_distribution(real_df_1, gen_df_2, n=10):
    """
    Computes the Jensen-Shannon Divergence (JSD) between the source port distributions
    of the real and generated datasets. This comparison is port-specific (not agnostic).

    Args:
        real_df_1 (pd.DataFrame): Real dataset containing network packet data.
        gen_df_2 (pd.DataFrame): Generated dataset containing network packet data.

    Returns:
        float: Jensen-Shannon Divergence (JSD) value in [0, 1].
    """
    # Count packets per source port and normalize
    real_dist = real_df_1['srcport'].value_counts(normalize=True)
    gen_dist = gen_df_2['srcport'].value_counts(normalize=True)

    # Align indices to ensure both have same port list
    all_ports = set(real_dist.index).union(set(gen_dist.index))
    real_aligned = real_dist.reindex(all_ports, fill_value=0).sort_index()
    gen_aligned = gen_dist.reindex(all_ports, fill_value=0).sort_index()

    # Compute and return JSD
    return jensenshannon_wrapper(real_aligned.values, gen_aligned.values, base=2)


def packet_stateless_dstport_countdistinct(real_df_1, gen_df_2, n=10):
    """
    Computes the Absolute Relative Error (ARE) of the number of distinct destination ports
    between the real and generated datasets.

    Args:
        real_df_1 (pd.DataFrame): Real dataset containing network packet data.
        gen_df_2 (pd.DataFrame): Generated dataset containing network packet data.

    Returns:
        float: Absolute Relative Error (ARE), capped at 1.0.
    """
    real_count = real_df_1['dstport'].nunique()
    gen_count = gen_df_2['dstport'].nunique()

    if real_count == 0:
        return 1.0 if gen_count > 0 else 0.0

    error = abs(real_count - gen_count) / real_count
    return min(error, 1.0)


def packet_stateless_dstport_distribution(real_df_1, gen_df_2, n=10):
    """
    Computes the Jensen-Shannon Divergence (JSD) between the destination port distributions
    of the real and generated datasets. This comparison is port-specific (not agnostic).

    Args:
        real_df_1 (pd.DataFrame): Real dataset containing network packet data.
        gen_df_2 (pd.DataFrame): Generated dataset containing network packet data.

    Returns:
        float: Jensen-Shannon Divergence (JSD) value in [0, 1].
    """
    from scipy.spatial.distance import jensenshannon

    # Count packets per destination port and normalize
    real_dist = real_df_1['dstport'].value_counts(normalize=True)
    gen_dist = gen_df_2['dstport'].value_counts(normalize=True)

    # Align indices to ensure both have the same set of ports
    all_ports = set(real_dist.index).union(set(gen_dist.index))
    real_aligned = real_dist.reindex(all_ports, fill_value=0).sort_index()
    gen_aligned = gen_dist.reindex(all_ports, fill_value=0).sort_index()

    # Compute and return JSD
    return jensenshannon_wrapper(real_aligned.values, gen_aligned.values, base=2)


def packet_stateless_proto_countdistinct(real_df_1, gen_df_2, n=10):
    """
    Computes the Absolute Relative Error (ARE) of the number of distinct protocols
    between the real and generated datasets.

    Args:
        real_df_1 (pd.DataFrame): Real dataset containing network packet data.
        gen_df_2 (pd.DataFrame): Generated dataset containing network packet data.

    Returns:
        float: Absolute Relative Error (ARE), capped at 1.0.
    """
    real_count = real_df_1['proto'].nunique()
    gen_count = gen_df_2['proto'].nunique()

    if real_count == 0:
        return 1.0 if gen_count > 0 else 0.0

    error = abs(real_count - gen_count) / real_count
    return min(error, 1.0)


def packet_stateless_proto_distribution(real_df_1, gen_df_2, n=10):
    """
    Computes the Jensen-Shannon Divergence (JSD) between the protocol distributions
    of the real and generated datasets. This comparison is protocol-specific (not agnostic).

    Args:
        real_df_1 (pd.DataFrame): Real dataset containing network packet data.
        gen_df_2 (pd.DataFrame): Generated dataset containing network packet data.

    Returns:
        float: Jensen-Shannon Divergence (JSD) value in [0, 1].
    """
    from scipy.spatial.distance import jensenshannon

    # Count packets per protocol and normalize
    real_dist = real_df_1['proto'].value_counts(normalize=True)
    gen_dist = gen_df_2['proto'].value_counts(normalize=True)

    # Align indices to ensure both have the same set of protocols
    all_protocols = set(real_dist.index).union(set(gen_dist.index))
    real_aligned = real_dist.reindex(all_protocols, fill_value=0).sort_index()
    gen_aligned = gen_dist.reindex(all_protocols, fill_value=0).sort_index()

    # Compute and return JSD
    return jensenshannon_wrapper(real_aligned.values, gen_aligned.values, base=2)


def packet_stateless_time_distribution(real_df_1, gen_df_2, n=10):
    """
    Computes the Wasserstein distance (Earth Mover's Distance) between the normalized
    packet timestamps of the real and generated datasets.

    The timestamps are first mapped to the [0, 1] interval before computing the distance.

    Args:
        real_df_1 (pd.DataFrame): Real dataset containing network packet data.
        gen_df_2 (pd.DataFrame): Generated dataset containing network packet data.

    Returns:
        float: Wasserstein distance (p=1) between the two normalized timestamp distributions.
    """
    

    # Combine real and generated time columns
    combined_times = pd.concat([real_df_1["time"], gen_df_2["time"]])
    min_val = combined_times.min()
    max_val = combined_times.max()
    if max_val > min_val:
        normalized_combined = (combined_times - min_val) / (max_val - min_val)
    else:
        normalized_combined = combined_times * 0

    # Split normalized times back into real and generated parts
    real_norm = normalized_combined.iloc[:len(real_df_1)]
    gen_norm = normalized_combined.iloc[len(real_df_1):]

    return wasserstein_distance_wrapper(real_norm, gen_norm)


def packet_stateless_pktlen_sum(real_df_1, gen_df_2, n=10):
    """
    Computes the Absolute Relative Error (ARE) between the total packet lengths
    of the real and generated datasets.
    
    Args:
        real_df_1 (pd.DataFrame): Real dataset containing network packet data.
        gen_df_2 (pd.DataFrame): Generated dataset containing network packet data.
    
    Returns:
        float: Absolute Relative Error (ARE) of total packet lengths, capped at 1.0.
    """
    real_sum = real_df_1['pkt_len'].sum()
    gen_sum = gen_df_2['pkt_len'].sum()
    
    if real_sum == 0:
        return 1.0 if gen_sum > 0 else 0.0
    
    error = abs(real_sum - gen_sum) / real_sum
    return min(error, 1.0)


def packet_stateless_pktlen_avg(real_df_1, gen_df_2, n=10):
    """
    Computes the Absolute Relative Error (ARE) between the average packet lengths
    of the real and generated datasets.

    Args:
        real_df_1 (pd.DataFrame): Real dataset containing network packet data.
        gen_df_2 (pd.DataFrame): Generated dataset containing network packet data.

    Returns:
        float: Absolute Relative Error (ARE) of average packet lengths, capped at 1.0.
    """
    if len(real_df_1) == 0:
        return 1.0 if len(gen_df_2) > 0 else 0.0

    real_avg = real_df_1['pkt_len'].mean()
    gen_avg = gen_df_2['pkt_len'].mean()

    if real_avg == 0:
        return 1.0 if gen_avg > 0 else 0.0

    error = abs(real_avg - gen_avg) / real_avg
    return min(error, 1.0)


def packet_stateless_pktlen_distribution(real_df_1, gen_df_2, n=10):
    """
    Computes the Jensen-Shannon Divergence (JSD) between the packet length
    distributions of the real and generated datasets.

    Args:
        real_df_1 (pd.DataFrame): Real dataset containing network packet data.
        gen_df_2 (pd.DataFrame): Generated dataset containing network packet data.

    Returns:
        float: Jensen-Shannon Divergence (JSD) value in [0, 1].
    """
    from scipy.spatial.distance import jensenshannon

    # Create a common bin range across both datasets
    min_len = min(real_df_1['pkt_len'].min(), gen_df_2['pkt_len'].min())
    max_len = max(real_df_1['pkt_len'].max(), gen_df_2['pkt_len'].max())
    bins = np.arange(min_len, max_len + 2)  # +2 to include max_len in binning

    # Compute normalized histograms
    real_hist, _ = np.histogram(real_df_1['pkt_len'], bins=bins, density=True)
    gen_hist, _ = np.histogram(gen_df_2['pkt_len'], bins=bins, density=True)

    # Pad histograms to ensure they are the same length
    max_len_hist = max(len(real_hist), len(gen_hist))
    real_padded = np.pad(real_hist, (0, max_len_hist - len(real_hist)))
    gen_padded = np.pad(gen_hist, (0, max_len_hist - len(gen_hist)))

    # Compute and return the Jensen-Shannon Divergence
    return jensenshannon_wrapper(real_padded, gen_padded, base=2)


def packet_stateless_flag_countdistinct(real_df_1, gen_df_2, n=10):
    """
    Computes the Absolute Relative Error (ARE) of the number of distinct TCP flag values
    between the real and generated datasets.
    
    Args:
        real_df_1 (pd.DataFrame): Real dataset containing network packet data.
        gen_df_2 (pd.DataFrame): Generated dataset containing network packet data.
    
    Returns:
        float: Absolute Relative Error (ARE), capped at 1.0.
    """
    real_count = real_df_1['flag'].nunique()
    gen_count = gen_df_2['flag'].nunique()
    
    if real_count == 0:
        return 1.0 if gen_count > 0 else 0.0
    
    error = abs(real_count - gen_count) / real_count
    return min(error, 1.0)


def packet_stateless_flag_distribution(real_df_1, gen_df_2, n=10):
    """
    Computes the Jensen-Shannon Divergence (JSD) between the TCP flag distributions
    of the real and generated datasets.

    Args:
        real_df_1 (pd.DataFrame): Real dataset containing network packet data.
        gen_df_2 (pd.DataFrame): Generated dataset containing network packet data.

    Returns:
        float: Jensen-Shannon Divergence (JSD) value in [0, 1].
    """
    from scipy.spatial.distance import jensenshannon

    # Count flags and normalize
    real_dist = real_df_1['flag'].value_counts(normalize=True)
    gen_dist = gen_df_2['flag'].value_counts(normalize=True)

    # Align indices to ensure both have the same set of flags
    all_flags = set(real_dist.index).union(set(gen_dist.index))
    real_aligned = real_dist.reindex(all_flags, fill_value=0).sort_index()
    gen_aligned = gen_dist.reindex(all_flags, fill_value=0).sort_index()

    # Compute and return JSD
    return jensenshannon_wrapper(real_aligned.values, gen_aligned.values, base=2)


def packet_stateless_ttl_avg(real_df_1, gen_df_2, n=10):
    """
    Computes the Absolute Relative Error (ARE) between the average TTL values
    of the real and generated datasets.

    Args:
        real_df_1 (pd.DataFrame): Real dataset containing network packet data.
        gen_df_2 (pd.DataFrame): Generated dataset containing network packet data.

    Returns:
        float: Absolute Relative Error (ARE) of average TTL values, capped at 1.0.
    """
    if len(real_df_1) == 0:
        return 1.0 if len(gen_df_2) > 0 else 0.0

    real_avg = real_df_1['ttl'].mean()
    gen_avg = gen_df_2['ttl'].mean()

    if real_avg == 0:
        return 1.0 if gen_avg > 0 else 0.0

    error = abs(real_avg - gen_avg) / real_avg
    return min(error, 1.0)


def packet_stateless_ttl_distribution(real_df_1, gen_df_2, n=10):
    """
    Computes the Jensen-Shannon Divergence (JSD) between the TTL distributions
    of the real and generated datasets.

    Args:
        real_df_1 (pd.DataFrame): Real dataset containing network packet data.
        gen_df_2 (pd.DataFrame): Generated dataset containing network packet data.

    Returns:
        float: Jensen-Shannon Divergence (JSD) value in [0, 1].
    """
    from scipy.spatial.distance import jensenshannon

    # Count TTL occurrences and normalize
    real_dist = real_df_1['ttl'].value_counts(normalize=True)
    gen_dist = gen_df_2['ttl'].value_counts(normalize=True)

    # Align indices to ensure both have the same TTL keys
    all_ttls = set(real_dist.index).union(set(gen_dist.index))
    real_aligned = real_dist.reindex(all_ttls, fill_value=0).sort_index()
    gen_aligned = gen_dist.reindex(all_ttls, fill_value=0).sort_index()

    # Compute and return JSD
    return jensenshannon_wrapper(real_aligned.values, gen_aligned.values, base=2)


def flow_srcip_stateless_packet_topnvalue(real_df_1, gen_df_2, n=10):
    """
    Computes the average Absolute Relative Error (ARE) of the number of packets sent by the top N source IPs
    between the real and generated datasets. The actual IP values are ignored; only the sorted top-N packet counts
    are compared.
    
    Args:
        real_df_1 (pd.DataFrame): Real dataset containing network packet data.
        gen_df_2 (pd.DataFrame): Generated dataset containing network packet data.
        n (int): Number of top source IPs to consider.
    
    Returns:
        float: Average ARE over the top-N source IPs, capped at 1.0.
    """
    # Aggregate packet count by srcip
    real_counts = real_df_1.groupby('srcip').size().nlargest(n).values
    gen_counts = gen_df_2.groupby('srcip').size().nlargest(n).values
    
    # Pad with zeros if one array is shorter
    max_len = max(len(real_counts), len(gen_counts))
    real_counts = np.pad(real_counts, (0, max_len - len(real_counts)))
    gen_counts = np.pad(gen_counts, (0, max_len - len(gen_counts)))
    
    # Compute relative errors
    relative_errors = np.abs(real_counts - gen_counts) / np.maximum(real_counts, 1)
    relative_errors = np.minimum(relative_errors, 1.0)
    
    # Compute mean ARE and cap at 1.0
    return min(relative_errors.mean(), 1.0)


def flow_srcip_stateless_bytes_topnvalue(real_df_1, gen_df_2, n=10):
    """
    Computes the average Absolute Relative Error (ARE) of the number of bytes sent by the top N source IPs
    between the real and generated datasets. The actual IP values are ignored; only the sorted top-N byte counts
    are compared.
    
    Args:
        real_df_1 (pd.DataFrame): Real dataset containing network packet data.
        gen_df_2 (pd.DataFrame): Generated dataset containing network packet data.
        n (int): Number of top source IPs to consider.
    
    Returns:
        float: Average ARE over the top-N source IPs, capped at 1.0.
    """
    # Aggregate byte count by srcip
    real_bytes = real_df_1.groupby('srcip')['pkt_len'].sum().nlargest(n).values
    gen_bytes = gen_df_2.groupby('srcip')['pkt_len'].sum().nlargest(n).values
 
    # Pad with zeros if one array is shorter
    max_len = max(len(real_bytes), len(gen_bytes))
    real_bytes = np.pad(real_bytes, (0, max_len - len(real_bytes)))
    gen_bytes = np.pad(gen_bytes, (0, max_len - len(gen_bytes)))
 
    # Compute relative errors
    relative_errors = np.abs(real_bytes - gen_bytes) / np.maximum(real_bytes, 1)
    relative_errors = np.minimum(relative_errors, 1.0)
 
    # Compute mean ARE and cap at 1.0
    return min(relative_errors.mean(), 1.0)


def flow_srcip_stateless_bytes_distribution(real_df_1, gen_df_2, n=10):
    """
    Computes the Jensen-Shannon Divergence (JSD) between the byte distributions
    of source IPs in the real and generated datasets. The IP addresses are ignored;
    only the sorted distribution values are compared (IP-agnostic).

    Args:
        real_df_1 (pd.DataFrame): Real dataset containing network packet data.
        gen_df_2 (pd.DataFrame): Generated dataset containing network packet data.

    Returns:
        float: Jensen-Shannon Divergence (JSD) value in [0, 1].
    """
    from scipy.spatial.distance import jensenshannon
    import numpy as np

    # Compute byte sums by srcip
    real_bytes = real_df_1.groupby('srcip')['pkt_len'].sum().values
    gen_bytes = gen_df_2.groupby('srcip')['pkt_len'].sum().values

    # Normalize to obtain probability distributions
    real_dist = real_bytes / real_bytes.sum() if real_bytes.sum() > 0 else np.zeros_like(real_bytes)
    gen_dist = gen_bytes / gen_bytes.sum() if gen_bytes.sum() > 0 else np.zeros_like(gen_bytes)

    # Sort distributions (IP-agnostic)
    real_sorted = np.sort(real_dist)[::-1]
    gen_sorted = np.sort(gen_dist)[::-1]

    # Pad to equal length
    max_len = max(len(real_sorted), len(gen_sorted))
    real_padded = np.pad(real_sorted, (0, max_len - len(real_sorted)))
    gen_padded = np.pad(gen_sorted, (0, max_len - len(gen_sorted)))

    return jensenshannon_wrapper(real_padded, gen_padded, base=2)


def flow_srcip_stateless_connection2srcport_topnvalue(real_df_1, gen_df_2, n=10):
    """
    Computes the average Absolute Relative Error (ARE) of the number of distinct source ports
    each of the top N source IPs connected to in the real and generated datasets.
    The IP addresses themselves are ignored; only the sorted top-N connection counts are compared.

    Args:
        real_df_1 (pd.DataFrame): Real dataset containing network packet data.
        gen_df_2 (pd.DataFrame): Generated dataset containing network packet data.
        n (int): Number of top source IPs to consider.

    Returns:
        float: Average ARE over the top-N source IPs, capped at 1.0.
    """
    # Count distinct srcport per srcip
    real_conn = real_df_1.groupby('srcip')['srcport'].nunique().nlargest(n).values
    gen_conn = gen_df_2.groupby('srcip')['srcport'].nunique().nlargest(n).values

    # Pad with zeros to equal length
    max_len = max(len(real_conn), len(gen_conn))
    real_conn = np.pad(real_conn, (0, max_len - len(real_conn)))
    gen_conn = np.pad(gen_conn, (0, max_len - len(gen_conn)))

    # Compute relative errors
    relative_errors = np.abs(real_conn - gen_conn) / np.maximum(real_conn, 1)
    relative_errors = np.minimum(relative_errors, 1.0)

    return min(relative_errors.mean(), 1.0)


def flow_srcip_stateless_connection2srcport_distribution(real_df_1, gen_df_2, n=10):
    """
    Computes the Jensen-Shannon Divergence (JSD) between the distribution of the number
    of distinct source ports contacted by each source IP in the real and generated datasets.
    The IP addresses are ignored; only the sorted distributions are compared.
 
    Args:
        real_df_1 (pd.DataFrame): Real dataset containing network packet data.
        gen_df_2 (pd.DataFrame): Generated dataset containing network packet data.
 
    Returns:
        float: Jensen-Shannon Divergence (JSD) value in [0, 1].
    """
    from scipy.spatial.distance import jensenshannon
    import numpy as np
 
    # Count distinct srcport per srcip
    real_conn = real_df_1.groupby('srcip')['srcport'].nunique().values
    gen_conn = gen_df_2.groupby('srcip')['srcport'].nunique().values
 
    # Normalize to obtain distributions
    real_dist = real_conn / real_conn.sum() if real_conn.sum() > 0 else np.zeros_like(real_conn)
    gen_dist = gen_conn / gen_conn.sum() if gen_conn.sum() > 0 else np.zeros_like(gen_conn)
 
    # Sort for IP-agnostic comparison
    real_sorted = np.sort(real_dist)[::-1]
    gen_sorted = np.sort(gen_dist)[::-1]
 
    # Pad to equal length
    max_len = max(len(real_sorted), len(gen_sorted))
    real_padded = np.pad(real_sorted, (0, max_len - len(real_sorted)))
    gen_padded = np.pad(gen_sorted, (0, max_len - len(gen_sorted)))
 
    return jensenshannon_wrapper(real_padded, gen_padded, base=2)


def flow_srcip_stateless_connection2dstip_topnvalue(real_df_1, gen_df_2, n=10):
    """
    Computes the average Absolute Relative Error (ARE) of the number of distinct destination IPs
    each of the top N source IPs (by number of connections) connected to in the real and generated datasets.
    The IP addresses themselves are ignored; only the sorted top-N connection counts are compared.

    Args:
        real_df_1 (pd.DataFrame): Real dataset containing network packet data.
        gen_df_2 (pd.DataFrame): Generated dataset containing network packet data.
        n (int): Number of top source IPs to consider.

    Returns:
        float: Average ARE over the top-N source IPs, capped at 1.0.
    """
    # Count distinct dstip per srcip
    real_conn = real_df_1.groupby('srcip')['dstip'].nunique().nlargest(n).values
    gen_conn = gen_df_2.groupby('srcip')['dstip'].nunique().nlargest(n).values

    # Pad with zeros to equal length
    max_len = max(len(real_conn), len(gen_conn))
    real_conn = np.pad(real_conn, (0, max_len - len(real_conn)))
    gen_conn = np.pad(gen_conn, (0, max_len - len(gen_conn)))

    # Compute relative errors
    relative_errors = np.abs(real_conn - gen_conn) / np.maximum(real_conn, 1)
    relative_errors = np.minimum(relative_errors, 1.0)

    return min(relative_errors.mean(), 1.0)


def flow_srcip_stateless_connection2dstip_distribution(real_df_1, gen_df_2, n=10):
    """
    Computes the Jensen-Shannon Divergence (JSD) between the distribution of the number
    of distinct destination IPs contacted by each source IP in the real and generated datasets.
    The IP addresses are ignored; only the sorted distributions are compared.

    Args:
        real_df_1 (pd.DataFrame): Real dataset containing network packet data.
        gen_df_2 (pd.DataFrame): Generated dataset containing network packet data.

    Returns:
        float: Jensen-Shannon Divergence (JSD) value in [0, 1].
    """
    from scipy.spatial.distance import jensenshannon
    import numpy as np

    # Count distinct dstip per srcip
    real_conn = real_df_1.groupby('srcip')['dstip'].nunique().values
    gen_conn = gen_df_2.groupby('srcip')['dstip'].nunique().values

    # Normalize to obtain distributions
    real_dist = real_conn / real_conn.sum() if real_conn.sum() > 0 else np.zeros_like(real_conn)
    gen_dist = gen_conn / gen_conn.sum() if gen_conn.sum() > 0 else np.zeros_like(gen_conn)

    # Sort for IP-agnostic comparison
    real_sorted = np.sort(real_dist)[::-1]
    gen_sorted = np.sort(gen_dist)[::-1]

    # Pad to equal length
    max_len = max(len(real_sorted), len(gen_sorted))
    real_padded = np.pad(real_sorted, (0, max_len - len(real_sorted)))
    gen_padded = np.pad(gen_sorted, (0, max_len - len(gen_sorted)))

    return jensenshannon_wrapper(real_padded, gen_padded, base=2)


def flow_srcip_stateless_connection2dstport_topnvalue(real_df_1, gen_df_2, n=10):
    """
    Computes the average Absolute Relative Error (ARE) of the number of distinct destination ports
    each of the top N source IPs connected to in the real and generated datasets.
    The IP addresses themselves are ignored; only the sorted top-N connection counts are compared.

    Args:
        real_df_1 (pd.DataFrame): Real dataset containing network packet data.
        gen_df_2 (pd.DataFrame): Generated dataset containing network packet data.
        n (int): Number of top source IPs to consider.

    Returns:
        float: Average ARE over the top-N source IPs, capped at 1.0.
    """
    # Count distinct dstport per srcip
    real_conn = real_df_1.groupby('srcip')['dstport'].nunique().nlargest(n).values
    gen_conn = gen_df_2.groupby('srcip')['dstport'].nunique().nlargest(n).values

    # Pad arrays to equal length
    max_len = max(len(real_conn), len(gen_conn))
    real_conn = np.pad(real_conn, (0, max_len - len(real_conn)))
    gen_conn = np.pad(gen_conn, (0, max_len - len(gen_conn)))

    # Compute relative errors
    relative_errors = np.abs(real_conn - gen_conn) / np.maximum(real_conn, 1)
    relative_errors = np.minimum(relative_errors, 1.0)

    return min(relative_errors.mean(), 1.0)


def flow_srcip_stateless_connection2dstport_distribution(real_df_1, gen_df_2, n=10):
    """
    Computes the Jensen-Shannon Divergence (JSD) between the distribution of the number
    of distinct destination ports contacted by each source IP in the real and generated datasets.
    The IP addresses are ignored; only the sorted distributions are compared.
 
    Args:
        real_df_1 (pd.DataFrame): Real dataset containing network packet data.
        gen_df_2 (pd.DataFrame): Generated dataset containing network packet data.
 
    Returns:
        float: Jensen-Shannon Divergence (JSD) value in [0, 1].
    """
    # Count distinct dstport per srcip
    real_conn = real_df_1.groupby('srcip')['dstport'].nunique().values
    gen_conn = gen_df_2.groupby('srcip')['dstport'].nunique().values
 
    # Normalize to obtain distributions
    real_dist = real_conn / real_conn.sum() if real_conn.sum() > 0 else np.zeros_like(real_conn)
    gen_dist = gen_conn / gen_conn.sum() if gen_conn.sum() > 0 else np.zeros_like(gen_conn)
 
    # Sort for IP-agnostic comparison
    real_sorted = np.sort(real_dist)[::-1]
    gen_sorted = np.sort(gen_dist)[::-1]
 
    # Pad to equal length
    max_len = max(len(real_sorted), len(gen_sorted))
    real_padded = np.pad(real_sorted, (0, max_len - len(real_sorted)))
    gen_padded = np.pad(gen_sorted, (0, max_len - len(gen_sorted)))
 
    return jensenshannon_wrapper(real_padded, gen_padded, base=2)


def flow_srcip_stateless_connection2dstipport_topnvalue(real_df_1, gen_df_2, n=10):
    """
    Computes the average Absolute Relative Error (ARE) of the number of distinct (dstip, dstport) pairs
    contacted by the top N source IPs between the real and generated datasets.
    The IP addresses themselves are ignored; only the sorted top-N counts are compared.

    Args:
        real_df_1 (pd.DataFrame): Real dataset containing network packet data.
        gen_df_2 (pd.DataFrame): Generated dataset containing network packet data.
        n (int): Number of top source IPs to consider.

    Returns:
        float: Average ARE over the top-N source IPs, capped at 1.0.
    """
    # Count distinct (dstip, dstport) per srcip
    real_conn = real_df_1.groupby('srcip').apply(lambda df: df[['dstip', 'dstport']].drop_duplicates().shape[0])
    gen_conn = gen_df_2.groupby('srcip').apply(lambda df: df[['dstip', 'dstport']].drop_duplicates().shape[0])

    real_top = real_conn.nlargest(n).values
    gen_top = gen_conn.nlargest(n).values

    # Pad with zeros to equal length
    max_len = max(len(real_top), len(gen_top))
    real_top = np.pad(real_top, (0, max_len - len(real_top)))
    gen_top = np.pad(gen_top, (0, max_len - len(gen_top)))

    # Compute relative errors
    relative_errors = np.abs(real_top - gen_top) / np.maximum(real_top, 1)
    relative_errors = np.minimum(relative_errors, 1.0)

    return min(relative_errors.mean(), 1.0)


def flow_srcip_stateless_connection2dstipport_distribution(real_df_1, gen_df_2, n=10):
    """
    Computes the Jensen-Shannon Divergence (JSD) between the distribution of the number
    of distinct (dstip, dstport) pairs contacted by each source IP in the real and generated datasets.
    The IP addresses are ignored; only the sorted distributions are compared.

    Args:
        real_df_1 (pd.DataFrame): Real dataset containing network packet data.
        gen_df_2 (pd.DataFrame): Generated dataset containing network packet data.

    Returns:
        float: Jensen-Shannon Divergence (JSD) value in [0, 1].
    """
    # Count distinct (dstip, dstport) pairs per srcip
    real_conn = real_df_1.groupby('srcip').apply(lambda df: df[['dstip', 'dstport']].drop_duplicates().shape[0]).values
    gen_conn = gen_df_2.groupby('srcip').apply(lambda df: df[['dstip', 'dstport']].drop_duplicates().shape[0]).values

    # Normalize to obtain probability distributions
    real_dist = real_conn / real_conn.sum() if real_conn.sum() > 0 else np.zeros_like(real_conn)
    gen_dist = gen_conn / gen_conn.sum() if gen_conn.sum() > 0 else np.zeros_like(gen_conn)

    # Sort for IP-agnostic comparison
    real_sorted = np.sort(real_dist)[::-1]
    gen_sorted = np.sort(gen_dist)[::-1]

    # Pad to equal length
    max_len = max(len(real_sorted), len(gen_sorted))
    real_padded = np.pad(real_sorted, (0, max_len - len(real_sorted)))
    gen_padded = np.pad(gen_sorted, (0, max_len - len(gen_sorted)))

    return jensenshannon_wrapper(real_padded, gen_padded, base=2)


def flow_srcip_stateless_connection2flow_topnvalue(real_df_1, gen_df_2, n=10):
    """
    Computes the average Absolute Relative Error (ARE) of the number of distinct flows
    (5-tuples: srcip, dstip, srcport, dstport, proto) per source IP for the top-N highest-flow source IPs.

    The IP addresses are ignored; only the top-N flow counts are compared.

    Args:
        real_df_1 (pd.DataFrame): Real dataset containing network packet data.
        gen_df_2 (pd.DataFrame): Generated dataset containing network packet data.
        n (int): Number of top source IPs to consider.

    Returns:
        float: Average ARE over the top-N source IPs, capped at 1.0.
    """
    def count_flows(df):
        # Create a full 5-tuple identifier for each packet
        df = df.copy()
        df['flow'] = list(zip(df['srcip'], df['dstip'], df['srcport'], df['dstport'], df['proto']))
        return df.groupby('srcip')['flow'].nunique()

    # Count flows per source IP
    real_counts = count_flows(real_df_1)
    gen_counts = count_flows(gen_df_2)

    # Extract top-N values
    real_top = real_counts.sort_values(ascending=False).head(n).values
    gen_top = gen_counts.sort_values(ascending=False).head(n).values

    # Pad arrays to equal length
    max_len = max(len(real_top), len(gen_top))
    real_top = np.pad(real_top, (0, max_len - len(real_top)))
    gen_top = np.pad(gen_top, (0, max_len - len(gen_top)))

    # Compute relative errors
    relative_errors = np.abs(real_top - gen_top) / np.maximum(real_top, 1)
    relative_errors = np.minimum(relative_errors, 1.0)

    return min(relative_errors.mean(), 1.0)


def flow_srcip_stateless_connection2flow_distribution(real_df_1, gen_df_2, n=10):
    """
    Computes the Jensen-Shannon Divergence (JSD) between the distribution of the number
    of distinct flows (5-tuples: srcip, dstip, srcport, dstport, proto) per source IP
    in the real and generated datasets.

    The IP addresses are ignored; only the sorted distributions are compared.

    Args:
        real_df_1 (pd.DataFrame): Real dataset containing network packet data.
        gen_df_2 (pd.DataFrame): Generated dataset containing network packet data.

    Returns:
        float: Jensen-Shannon Divergence (JSD) value in [0, 1].
    """
    def count_flows(df):
        df = df.copy()
        df['flow'] = list(zip(df['srcip'], df['dstip'], df['srcport'], df['dstport'], df['proto']))
        return df.groupby('srcip')['flow'].nunique().values

    # Count flows per srcip
    real_counts = count_flows(real_df_1)
    gen_counts = count_flows(gen_df_2)

    # Normalize to obtain probability distributions
    real_dist = real_counts / real_counts.sum() if real_counts.sum() > 0 else np.zeros_like(real_counts)
    gen_dist = gen_counts / gen_counts.sum() if gen_counts.sum() > 0 else np.zeros_like(gen_counts)

    # Sort for IP-agnostic comparison
    real_sorted = np.sort(real_dist)[::-1]
    gen_sorted = np.sort(gen_dist)[::-1]

    # Pad to equal length
    max_len = max(len(real_sorted), len(gen_sorted))
    real_padded = np.pad(real_sorted, (0, max_len - len(real_sorted)))
    gen_padded = np.pad(gen_sorted, (0, max_len - len(gen_sorted)))

    return jensenshannon_wrapper(real_padded, gen_padded, base=2)


def flow_srcip_stateful_avgpacketinterval_topnvalue(real_df_1, gen_df_2, n=10):
    """
    Computes the average Absolute Relative Error (ARE) of the average packet interval
    for the top N source IPs (by number of packets) between the real and generated datasets.
    IP addresses are ignored; only the sorted average intervals are compared.
 
    Args:
        real_df_1 (pd.DataFrame): Real dataset containing network packet data.
        gen_df_2 (pd.DataFrame): Generated dataset containing network packet data.
        n (int): Number of top source IPs to consider.
 
    Returns:
        float: Average ARE over the top-N average intervals, capped at 1.0.
    """
    def compute_avg_intervals(df):
        valid = df.groupby('srcip').filter(lambda x: len(x) > 1)
        intervals = valid.sort_values('time').groupby('srcip')['time'].agg(lambda x: np.diff(x).mean())
        return intervals
 
    real_intervals = compute_avg_intervals(real_df_1).nlargest(n).values
    gen_intervals = compute_avg_intervals(gen_df_2).nlargest(n).values
 
    # Pad with zeros to equal length
    max_len = max(len(real_intervals), len(gen_intervals))
    real_intervals = np.pad(real_intervals, (0, max_len - len(real_intervals)))
    gen_intervals = np.pad(gen_intervals, (0, max_len - len(gen_intervals)))
 
    # Compute relative errors
    relative_errors = np.abs(real_intervals - gen_intervals) / np.maximum(real_intervals, 1)
    relative_errors = np.minimum(relative_errors, 1.0)
 
    return min(relative_errors.mean(), 1.0)

def flow_srcip_stateful_avgpacketinterval_distribution(real_df_1, gen_df_2, n=10):
    """
    Computes the Jensen-Shannon Divergence (JSD) between the distributions
    of average packet intervals for each source IP in the real and generated datasets.
    IP addresses are ignored; only the sorted distributions are compared.

    Args:
        real_df_1 (pd.DataFrame): Real dataset containing network packet data.
        gen_df_2 (pd.DataFrame): Generated dataset containing network packet data.

    Returns:
        float: Jensen-Shannon Divergence (JSD) in [0, 1].
    """
    import numpy as np

    def compute_avg_intervals(df):
        valid = df.groupby('srcip').filter(lambda x: len(x) > 1)
        return valid.sort_values('time').groupby('srcip')['time'].agg(lambda x: np.diff(x).mean()).values

    # Compute average intervals
    real_avg_intervals = compute_avg_intervals(real_df_1)
    gen_avg_intervals = compute_avg_intervals(gen_df_2)

    # Convert to distributions
    real_dist = real_avg_intervals / real_avg_intervals.sum() if real_avg_intervals.sum() > 0 else np.zeros_like(real_avg_intervals)
    gen_dist = gen_avg_intervals / gen_avg_intervals.sum() if gen_avg_intervals.sum() > 0 else np.zeros_like(gen_avg_intervals)

    # Sort for flow-agnostic comparison
    real_sorted = np.sort(real_dist)[::-1]
    gen_sorted = np.sort(gen_dist)[::-1]

    # Use the provided wrapper
    return jensenshannon_wrapper(real_sorted, gen_sorted, base=2)


def flow_srcip_stateful_flowduration_topnvalue(real_df_1, gen_df_2, n=10):
    """
    Computes the average Absolute Relative Error (ARE) of the flow durations
    for the top N source IPs between the real and generated datasets.
    IP addresses are ignored; only the sorted top-N flow durations are compared.

    A flow's duration is defined as the time difference between the first and last
    packets associated with each srcip.

    Args:
        real_df_1 (pd.DataFrame): Real dataset containing network packet data.
        gen_df_2 (pd.DataFrame): Generated dataset containing network packet data.
        n (int): Number of top source IPs to consider.

    Returns:
        float: Average ARE over the top-N flow durations, capped at 1.0.
    """
    def compute_durations(df):
        grouped = df.groupby('srcip')['time']
        return (grouped.max() - grouped.min()).values

    real_durations = compute_durations(real_df_1)
    gen_durations = compute_durations(gen_df_2)

    real_top = np.sort(real_durations)[-n:][::-1]
    gen_top = np.sort(gen_durations)[-n:][::-1]

    max_len = max(len(real_top), len(gen_top))
    real_top = np.pad(real_top, (0, max_len - len(real_top)))
    gen_top = np.pad(gen_top, (0, max_len - len(gen_top)))

    relative_errors = np.abs(real_top - gen_top) / np.maximum(real_top, 1)
    relative_errors = np.minimum(relative_errors, 1.0)

    return min(relative_errors.mean(), 1.0)


def flow_srcip_stateful_flowduration_distribution(real_df_1, gen_df_2, n=10):
    """
    Computes the Jensen-Shannon Divergence (JSD) between the distributions
    of flow durations for each source IP in the real and generated datasets.
    IP addresses are ignored; only the sorted distributions are compared.

    A flow's duration is defined as the time difference between the first and last
    packets associated with each srcip.

    Args:
        real_df_1 (pd.DataFrame): Real dataset containing network packet data.
        gen_df_2 (pd.DataFrame): Generated dataset containing network packet data.

    Returns:
        float: Jensen-Shannon Divergence (JSD) in [0, 1].
    """
    import numpy as np

    def compute_durations(df):
        grouped = df.groupby('srcip')['time']
        return (grouped.max() - grouped.min()).values

    real_durations = compute_durations(real_df_1)
    gen_durations = compute_durations(gen_df_2)

    # Convert to probability distributions
    real_dist = real_durations / real_durations.sum() if real_durations.sum() > 0 else np.zeros_like(real_durations)
    gen_dist = gen_durations / gen_durations.sum() if gen_durations.sum() > 0 else np.zeros_like(gen_durations)

    # Sort for srcip-agnostic comparison
    real_sorted = np.sort(real_dist)[::-1]
    gen_sorted = np.sort(gen_dist)[::-1]

    return jensenshannon_wrapper(real_sorted, gen_sorted, base=2)


def flow_srcip_stateful_byterate_topnvalue(real_df_1, gen_df_2, n=10):
    """
    Computes the average Absolute Relative Error (ARE) of byte rates (bytes per microsecond)
    for the top N source IPs in the real and generated datasets. The actual IP values are ignored;
    only the sorted top-N byte rates are compared.

    Args:
        real_df_1 (pd.DataFrame): Real dataset containing network packet data.
        gen_df_2 (pd.DataFrame): Generated dataset containing network packet data.
        n (int): Number of top source IPs to consider.

    Returns:
        float: Average ARE over the top-N byte rates, capped at 1.0.
    """
    def compute_byterates(df):
        grouped = df.groupby('srcip')
        valid_srcips = grouped.filter(lambda x: len(x) > 1).groupby('srcip')
        bytes_sent = valid_srcips['pkt_len'].sum()
        duration = valid_srcips['time'].agg(lambda x: x.max() - x.min())
        duration[duration == 0] = 1  # prevent division by zero
        return (bytes_sent / duration)

    real_rates = compute_byterates(real_df_1).nlargest(n).values
    gen_rates = compute_byterates(gen_df_2).nlargest(n).values

    # Pad arrays to equal length
    max_len = max(len(real_rates), len(gen_rates))
    real_rates = np.pad(real_rates, (0, max_len - len(real_rates)))
    gen_rates = np.pad(gen_rates, (0, max_len - len(gen_rates)))

    # Compute relative errors
    relative_errors = np.abs(real_rates - gen_rates) / np.maximum(real_rates, 1)
    relative_errors = np.minimum(relative_errors, 1.0)

    return min(relative_errors.mean(), 1.0)


def flow_srcip_stateful_byterate_distribution(real_df_1, gen_df_2, n=10):
    """
    Computes the Jensen-Shannon Divergence (JSD) between the distributions
    of byte rates (bytes per microsecond) for each source IP in the real and generated datasets.
    IP addresses are ignored; only the sorted distributions are compared.

    Args:
        real_df_1 (pd.DataFrame): Real dataset containing network packet data.
        gen_df_2 (pd.DataFrame): Generated dataset containing network packet data.

    Returns:
        float: Jensen-Shannon Divergence (JSD) in [0, 1].
    """
    import numpy as np

    def compute_byterates(df):
        grouped = df.groupby('srcip')
        valid_srcips = grouped.filter(lambda x: len(x) > 1).groupby('srcip')
        bytes_sent = valid_srcips['pkt_len'].sum()
        duration = valid_srcips['time'].agg(lambda x: x.max() - x.min())
        duration[duration == 0] = 1  # avoid division by zero
        return (bytes_sent / duration).values

    real_rates = compute_byterates(real_df_1)
    gen_rates = compute_byterates(gen_df_2)

    # Convert to distributions
    real_dist = real_rates / real_rates.sum() if real_rates.sum() > 0 else np.zeros_like(real_rates)
    gen_dist = gen_rates / gen_rates.sum() if gen_rates.sum() > 0 else np.zeros_like(gen_rates)

    # Sort for srcip-agnostic comparison
    real_sorted = np.sort(real_dist)[::-1]
    gen_sorted = np.sort(gen_dist)[::-1]

    return jensenshannon_wrapper(real_sorted, gen_sorted, base=2)


def flow_dstip_stateless_packet_topnvalue(real_df_1, gen_df_2, n=10):
    """
    Computes the average Absolute Relative Error (ARE) of the number of packets received by the top N destination IPs
    between the real and generated datasets. The actual IP values are ignored; only the sorted top-N packet counts
    are compared.

    Args:
        real_df_1 (pd.DataFrame): Real dataset containing network packet data.
        gen_df_2 (pd.DataFrame): Generated dataset containing network packet data.
        n (int): Number of top destination IPs to consider.

    Returns:
        float: Average ARE over the top-N destination IPs, capped at 1.0.
    """
    # Aggregate packet count by dstip
    real_counts = real_df_1.groupby('dstip').size().nlargest(n).values
    gen_counts = gen_df_2.groupby('dstip').size().nlargest(n).values

    # Pad with zeros if one array is shorter
    max_len = max(len(real_counts), len(gen_counts))
    real_counts = np.pad(real_counts, (0, max_len - len(real_counts)))
    gen_counts = np.pad(gen_counts, (0, max_len - len(gen_counts)))

    # Compute relative errors
    relative_errors = np.abs(real_counts - gen_counts) / np.maximum(real_counts, 1)
    relative_errors = np.minimum(relative_errors, 1.0)

    # Compute mean ARE and cap at 1.0
    return min(relative_errors.mean(), 1.0)


def flow_dstip_stateless_bytes_topnvalue(real_df_1, gen_df_2, n=10):
    """
    Computes the average Absolute Relative Error (ARE) of the number of bytes received by the top N destination IPs
    between the real and generated datasets. The actual IP values are ignored; only the sorted top-N byte counts
    are compared.

    Args:
        real_df_1 (pd.DataFrame): Real dataset containing network packet data.
        gen_df_2 (pd.DataFrame): Generated dataset containing network packet data.
        n (int): Number of top destination IPs to consider.

    Returns:
        float: Average ARE over the top-N destination IPs, capped at 1.0.
    """
    # Aggregate byte count by dstip
    real_bytes = real_df_1.groupby('dstip')['pkt_len'].sum().nlargest(n).values
    gen_bytes = gen_df_2.groupby('dstip')['pkt_len'].sum().nlargest(n).values

    # Pad with zeros if one array is shorter
    max_len = max(len(real_bytes), len(gen_bytes))
    real_bytes = np.pad(real_bytes, (0, max_len - len(real_bytes)))
    gen_bytes = np.pad(gen_bytes, (0, max_len - len(gen_bytes)))

    # Compute relative errors
    relative_errors = np.abs(real_bytes - gen_bytes) / np.maximum(real_bytes, 1)
    relative_errors = np.minimum(relative_errors, 1.0)

    # Compute mean ARE and cap at 1.0
    return min(relative_errors.mean(), 1.0)

def flow_dstip_stateless_bytes_distribution(real_df_1, gen_df_2, n=10):
    """
    Computes the Jensen-Shannon Divergence (JSD) between the byte distributions
    of destination IPs in the real and generated datasets. The IP addresses are ignored;
    only the sorted distribution values are compared (IP-agnostic).

    Args:
        real_df_1 (pd.DataFrame): Real dataset containing network packet data.
        gen_df_2 (pd.DataFrame): Generated dataset containing network packet data.

    Returns:
        float: Jensen-Shannon Divergence (JSD) value in [0, 1].
    """
    from scipy.spatial.distance import jensenshannon
    import numpy as np

    # Compute byte sums by dstip
    real_bytes = real_df_1.groupby('dstip')['pkt_len'].sum().values
    gen_bytes = gen_df_2.groupby('dstip')['pkt_len'].sum().values

    # Normalize to obtain probability distributions
    real_dist = real_bytes / real_bytes.sum() if real_bytes.sum() > 0 else np.zeros_like(real_bytes)
    gen_dist = gen_bytes / gen_bytes.sum() if gen_bytes.sum() > 0 else np.zeros_like(gen_bytes)

    # Sort distributions (IP-agnostic)
    real_sorted = np.sort(real_dist)[::-1]
    gen_sorted = np.sort(gen_dist)[::-1]

    # Pad to equal length
    max_len = max(len(real_sorted), len(gen_sorted))
    real_padded = np.pad(real_sorted, (0, max_len - len(real_sorted)))
    gen_padded = np.pad(gen_sorted, (0, max_len - len(gen_sorted)))

    return jensenshannon_wrapper(real_padded, gen_padded, base=2)


def flow_dstip_stateless_connection2dstport_topnvalue(real_df_1, gen_df_2, n=10):
    """
    Computes the average Absolute Relative Error (ARE) of the number of distinct destination ports
    each of the top N destination IPs connected to in the real and generated datasets.
    The IP addresses themselves are ignored; only the sorted top-N connection counts are compared.

    Args:
        real_df_1 (pd.DataFrame): Real dataset containing network packet data.
        gen_df_2 (pd.DataFrame): Generated dataset containing network packet data.
        n (int): Number of top destination IPs to consider.

    Returns:
        float: Average ARE over the top-N destination IPs, capped at 1.0.
    """
    # Count distinct dstport per dstip
    real_conn = real_df_1.groupby('dstip')['dstport'].nunique().nlargest(n).values
    gen_conn = gen_df_2.groupby('dstip')['dstport'].nunique().nlargest(n).values

    # Pad with zeros to equal length
    max_len = max(len(real_conn), len(gen_conn))
    real_conn = np.pad(real_conn, (0, max_len - len(real_conn)))
    gen_conn = np.pad(gen_conn, (0, max_len - len(gen_conn)))

    # Compute relative errors
    relative_errors = np.abs(real_conn - gen_conn) / np.maximum(real_conn, 1)
    relative_errors = np.minimum(relative_errors, 1.0)

    return min(relative_errors.mean(), 1.0)


def flow_dstip_stateless_connection2dstport_distribution(real_df_1, gen_df_2, n=10):
    """
    Computes the Jensen-Shannon Divergence (JSD) between the distribution of the number
    of distinct destination ports contacted by each destination IP in the real and generated datasets.
    The IP addresses are ignored; only the sorted distributions are compared.

    Args:
        real_df_1 (pd.DataFrame): Real dataset containing network packet data.
        gen_df_2 (pd.DataFrame): Generated dataset containing network packet data.

    Returns:
        float: Jensen-Shannon Divergence (JSD) value in [0, 1].
    """
    from scipy.spatial.distance import jensenshannon
    import numpy as np

    # Count distinct dstport per dstip
    real_conn = real_df_1.groupby('dstip')['dstport'].nunique().values
    gen_conn = gen_df_2.groupby('dstip')['dstport'].nunique().values

    # Normalize to obtain distributions
    real_dist = real_conn / real_conn.sum() if real_conn.sum() > 0 else np.zeros_like(real_conn)
    gen_dist = gen_conn / gen_conn.sum() if gen_conn.sum() > 0 else np.zeros_like(gen_conn)

    # Sort for IP-agnostic comparison
    real_sorted = np.sort(real_dist)[::-1]
    gen_sorted = np.sort(gen_dist)[::-1]

    # Pad to equal length
    max_len = max(len(real_sorted), len(gen_sorted))
    real_padded = np.pad(real_sorted, (0, max_len - len(real_sorted)))
    gen_padded = np.pad(gen_sorted, (0, max_len - len(gen_sorted)))

    return jensenshannon_wrapper(real_padded, gen_padded, base=2)


def flow_dstip_stateless_connection2srcip_topnvalue(real_df_1, gen_df_2, n=10):
    """
    Computes the average Absolute Relative Error (ARE) of the number of distinct source IPs
    each of the top N destination IPs connected to in the real and generated datasets.
    The IP addresses themselves are ignored; only the sorted top-N connection counts are compared.

    Args:
        real_df_1 (pd.DataFrame): Real dataset containing network packet data.
        gen_df_2 (pd.DataFrame): Generated dataset containing network packet data.
        n (int): Number of top destination IPs to consider.

    Returns:
        float: Average ARE over the top-N destination IPs, capped at 1.0.
    """
    # Count distinct srcip per dstip
    real_conn = real_df_1.groupby('dstip')['srcip'].nunique().nlargest(n).values
    gen_conn = gen_df_2.groupby('dstip')['srcip'].nunique().nlargest(n).values

    # Pad arrays to equal length
    max_len = max(len(real_conn), len(gen_conn))
    real_conn = np.pad(real_conn, (0, max_len - len(real_conn)))
    gen_conn = np.pad(gen_conn, (0, max_len - len(gen_conn)))

    # Compute relative errors
    relative_errors = np.abs(real_conn - gen_conn) / np.maximum(real_conn, 1)
    relative_errors = np.minimum(relative_errors, 1.0)

    return min(relative_errors.mean(), 1.0)


def flow_dstip_stateless_connection2srcip_distribution(real_df_1, gen_df_2, n=10):
    """
    Computes the Jensen-Shannon Divergence (JSD) between the distribution of the number
    of distinct source IPs connected to each destination IP in the real and generated datasets.
    The IP addresses are ignored; only the sorted distributions are compared.

    Args:
        real_df_1 (pd.DataFrame): Real dataset containing network packet data.
        gen_df_2 (pd.DataFrame): Generated dataset containing network packet data.

    Returns:
        float: Jensen-Shannon Divergence (JSD) value in [0, 1].
    """
    from scipy.spatial.distance import jensenshannon
    import numpy as np

    # Count distinct srcip per dstip
    real_conn = real_df_1.groupby('dstip')['srcip'].nunique().values
    gen_conn = gen_df_2.groupby('dstip')['srcip'].nunique().values

    # Normalize to obtain distributions
    real_dist = real_conn / real_conn.sum() if real_conn.sum() > 0 else np.zeros_like(real_conn)
    gen_dist = gen_conn / gen_conn.sum() if gen_conn.sum() > 0 else np.zeros_like(gen_conn)

    # Sort for IP-agnostic comparison
    real_sorted = np.sort(real_dist)[::-1]
    gen_sorted = np.sort(gen_dist)[::-1]

    # Pad to equal length
    max_len = max(len(real_sorted), len(gen_sorted))
    real_padded = np.pad(real_sorted, (0, max_len - len(real_sorted)))
    gen_padded = np.pad(gen_sorted, (0, max_len - len(gen_sorted)))

    return jensenshannon_wrapper(real_padded, gen_padded, base=2)


def flow_dstip_stateless_connection2srcport_topnvalue(real_df_1, gen_df_2, n=10):
    """
    Computes the average Absolute Relative Error (ARE) of the number of distinct source ports
    each of the top N destination IPs connected to in the real and generated datasets.
    The IP addresses themselves are ignored; only the sorted top-N connection counts are compared.

    Args:
        real_df_1 (pd.DataFrame): Real dataset containing network packet data.
        gen_df_2 (pd.DataFrame): Generated dataset containing network packet data.
        n (int): Number of top destination IPs to consider.

    Returns:
        float: Average ARE over the top-N destination IPs, capped at 1.0.
    """
    import numpy as np

    # Count distinct srcport per dstip
    real_conn = real_df_1.groupby('dstip')['srcport'].nunique().nlargest(n).values
    gen_conn = gen_df_2.groupby('dstip')['srcport'].nunique().nlargest(n).values

    # Pad with zeros to equal length
    max_len = max(len(real_conn), len(gen_conn))
    real_conn = np.pad(real_conn, (0, max_len - len(real_conn)))
    gen_conn = np.pad(gen_conn, (0, max_len - len(gen_conn)))

    # Compute relative errors
    relative_errors = np.abs(real_conn - gen_conn) / np.maximum(real_conn, 1)
    relative_errors = np.minimum(relative_errors, 1.0)

    return min(relative_errors.mean(), 1.0)


def flow_dstip_stateless_connection2srcport_distribution(real_df_1, gen_df_2, n=10):
    """
    Computes the Jensen-Shannon Divergence (JSD) between the distribution of the number
    of distinct source ports connected to each destination IP in the real and generated datasets.
    The IP addresses are ignored; only the sorted distributions are compared.
 
    Args:
        real_df_1 (pd.DataFrame): Real dataset containing network packet data.
        gen_df_2 (pd.DataFrame): Generated dataset containing network packet data.
 
    Returns:
        float: Jensen-Shannon Divergence (JSD) value in [0, 1].
    """
    from scipy.spatial.distance import jensenshannon
    import numpy as np
 
    # Count distinct srcport per dstip
    real_conn = real_df_1.groupby('dstip')['srcport'].nunique().values
    gen_conn = gen_df_2.groupby('dstip')['srcport'].nunique().values
 
    # Normalize to obtain probability distributions
    real_dist = real_conn / real_conn.sum() if real_conn.sum() > 0 else np.zeros_like(real_conn)
    gen_dist = gen_conn / gen_conn.sum() if gen_conn.sum() > 0 else np.zeros_like(gen_conn)
 
    # Sort for IP-agnostic comparison
    real_sorted = np.sort(real_dist)[::-1]
    gen_sorted = np.sort(gen_dist)[::-1]
 
    # Pad to equal length
    max_len = max(len(real_sorted), len(gen_sorted))
    real_padded = np.pad(real_sorted, (0, max_len - len(real_sorted)))
    gen_padded = np.pad(gen_sorted, (0, max_len - len(gen_sorted)))
 
    return jensenshannon_wrapper(real_padded, gen_padded, base=2)


def flow_dstip_stateless_connection2srcipport_topnvalue(real_df_1, gen_df_2, n=10):
    """
    Computes the average Absolute Relative Error (ARE) of the number of distinct (srcip, srcport) pairs
    connected to each of the top N destination IPs between the real and generated datasets.
    The IP addresses themselves are ignored; only the sorted top-N counts are compared.

    Args:
        real_df_1 (pd.DataFrame): Real dataset containing network packet data.
        gen_df_2 (pd.DataFrame): Generated dataset containing network packet data.
        n (int): Number of top destination IPs to consider.

    Returns:
        float: Average ARE over the top-N destination IPs, capped at 1.0.
    """
    # Count distinct (srcip, srcport) pairs per dstip
    real_conn = real_df_1.groupby('dstip').apply(lambda df: df[['srcip', 'srcport']].drop_duplicates().shape[0])
    gen_conn = gen_df_2.groupby('dstip').apply(lambda df: df[['srcip', 'srcport']].drop_duplicates().shape[0])

    # Take the top N values
    real_top = real_conn.nlargest(n).values
    gen_top = gen_conn.nlargest(n).values

    # Pad with zeros to equal length
    max_len = max(len(real_top), len(gen_top))
    real_top = np.pad(real_top, (0, max_len - len(real_top)))
    gen_top = np.pad(gen_top, (0, max_len - len(gen_top)))

    # Compute relative errors
    relative_errors = np.abs(real_top - gen_top) / np.maximum(real_top, 1)
    relative_errors = np.minimum(relative_errors, 1.0)

    return min(relative_errors.mean(), 1.0)


def flow_dstip_stateless_connection2srcipport_distribution(real_df_1, gen_df_2, n=10):
    """
    Computes the Jensen-Shannon Divergence (JSD) between the distribution of the number
    of distinct (srcip, srcport) pairs connected to each destination IP in the real and generated datasets.
    The IP addresses are ignored; only the sorted distributions are compared.

    Args:
        real_df_1 (pd.DataFrame): Real dataset containing network packet data.
        gen_df_2 (pd.DataFrame): Generated dataset containing network packet data.

    Returns:
        float: Jensen-Shannon Divergence (JSD) value in [0, 1].
    """
    from scipy.spatial.distance import jensenshannon
    import numpy as np

    # Count distinct (srcip, srcport) pairs per dstip
    real_conn = real_df_1.groupby('dstip').apply(lambda df: df[['srcip', 'srcport']].drop_duplicates().shape[0]).values
    gen_conn = gen_df_2.groupby('dstip').apply(lambda df: df[['srcip', 'srcport']].drop_duplicates().shape[0]).values

    # Normalize to obtain probability distributions
    real_dist = real_conn / real_conn.sum() if real_conn.sum() > 0 else np.zeros_like(real_conn)
    gen_dist = gen_conn / gen_conn.sum() if gen_conn.sum() > 0 else np.zeros_like(gen_conn)

    # Sort for IP-agnostic comparison
    real_sorted = np.sort(real_dist)[::-1]
    gen_sorted = np.sort(gen_dist)[::-1]

    # Pad to equal length
    max_len = max(len(real_sorted), len(gen_sorted))
    real_padded = np.pad(real_sorted, (0, max_len - len(real_sorted)))
    gen_padded = np.pad(gen_sorted, (0, max_len - len(gen_sorted)))

    return jensenshannon_wrapper(real_padded, gen_padded, base=2)


def flow_dstip_stateless_connection2flow_topnvalue(real_df_1, gen_df_2, n=10):
    """
    Computes the average Absolute Relative Error (ARE) of the number of distinct flows
    (5-tuples: srcip, dstip, srcport, dstport, proto) per destination IP for the top-N highest-flow destination IPs.

    The IP addresses are ignored; only the top-N flow counts are compared.

    Args:
        real_df_1 (pd.DataFrame): Real dataset containing network packet data.
        gen_df_2 (pd.DataFrame): Generated dataset containing network packet data.
        n (int): Number of top destination IPs to consider.

    Returns:
        float: Average ARE over the top-N destination IPs, capped at 1.0.
    """
    import numpy as np

    def count_flows(df):
        df = df.copy()
        df['flow'] = list(zip(df['srcip'], df['dstip'], df['srcport'], df['dstport'], df['proto']))
        return df.groupby('dstip')['flow'].nunique()

    real_counts = count_flows(real_df_1)
    gen_counts = count_flows(gen_df_2)

    real_top = real_counts.sort_values(ascending=False).head(n).values
    gen_top = gen_counts.sort_values(ascending=False).head(n).values

    max_len = max(len(real_top), len(gen_top))
    real_top = np.pad(real_top, (0, max_len - len(real_top)))
    gen_top = np.pad(gen_top, (0, max_len - len(gen_top)))

    relative_errors = np.abs(real_top - gen_top) / np.maximum(real_top, 1)
    relative_errors = np.minimum(relative_errors, 1.0)

    return min(relative_errors.mean(), 1.0)


def flow_dstip_stateless_connection2flow_distribution(real_df_1, gen_df_2, n=10):
    """
    Computes the Jensen-Shannon Divergence (JSD) between the distribution of the number
    of distinct flows (5-tuples: srcip, dstip, srcport, dstport, proto) contacted by each 
    destination IP in the real and generated datasets.

    The IP addresses are ignored; only the sorted distributions are compared.

    Args:
        real_df_1 (pd.DataFrame): Real dataset containing network packet data.
        gen_df_2 (pd.DataFrame): Generated dataset containing network packet data.

    Returns:
        float: Jensen-Shannon Divergence (JSD) value in [0, 1].
    """
    from scipy.spatial.distance import jensenshannon
    import numpy as np

    def count_flows(df):
        df = df.copy()
        df['flow'] = list(zip(df['srcip'], df['dstip'], df['srcport'], df['dstport'], df['proto']))
        return df.groupby('dstip')['flow'].nunique().values

    # Count flows per destination IP
    real_conn = count_flows(real_df_1)
    gen_conn = count_flows(gen_df_2)

    # Normalize to obtain probability distributions
    real_dist = real_conn / real_conn.sum() if real_conn.sum() > 0 else np.zeros_like(real_conn)
    gen_dist = gen_conn / gen_conn.sum() if gen_conn.sum() > 0 else np.zeros_like(gen_conn)

    # Sort for IP-agnostic comparison
    real_sorted = np.sort(real_dist)[::-1]
    gen_sorted = np.sort(gen_dist)[::-1]

    # Pad to equal length
    max_len = max(len(real_sorted), len(gen_sorted))
    real_padded = np.pad(real_sorted, (0, max_len - len(real_sorted)))
    gen_padded = np.pad(gen_sorted, (0, max_len - len(gen_sorted)))

    return jensenshannon_wrapper(real_padded, gen_padded, base=2)


def flow_dstip_stateful_avgpacketinterval_topnvalue(real_df_1, gen_df_2, n=10):
    """
    Computes the average Absolute Relative Error (ARE) of the average packet interval
    for the top N destination IPs (by number of packets) between the real and generated datasets.
    IP addresses are ignored; only the sorted average intervals are compared.

    Args:
        real_df_1 (pd.DataFrame): Real dataset containing network packet data.
        gen_df_2 (pd.DataFrame): Generated dataset containing network packet data.
        n (int): Number of top destination IPs to consider.

    Returns:
        float: Average ARE over the top-N average intervals, capped at 1.0.
    """
    import numpy as np

    def compute_avg_intervals(df):
        # Only consider flows with more than one packet to compute intervals
        valid = df.groupby('dstip').filter(lambda x: len(x) > 1)
        intervals = valid.sort_values('time').groupby('dstip')['time'].agg(lambda x: np.diff(x).mean())
        return intervals

    # Compute average intervals for real and generated datasets
    real_intervals = compute_avg_intervals(real_df_1).nlargest(n).values
    gen_intervals = compute_avg_intervals(gen_df_2).nlargest(n).values

    # Pad with zeros to ensure same length
    max_len = max(len(real_intervals), len(gen_intervals))
    real_intervals = np.pad(real_intervals, (0, max_len - len(real_intervals)))
    gen_intervals = np.pad(gen_intervals, (0, max_len - len(gen_intervals)))

    # Compute relative errors
    relative_errors = np.abs(real_intervals - gen_intervals) / np.maximum(real_intervals, 1)
    relative_errors = np.minimum(relative_errors, 1.0)

    return min(relative_errors.mean(), 1.0)


def flow_dstip_stateful_avgpacketinterval_distribution(real_df_1, gen_df_2, n=10):
    """
    Computes the Jensen-Shannon Divergence (JSD) between the distributions
    of average packet intervals for each destination IP in the real and generated datasets.
    IP addresses are ignored; only the sorted distributions are compared.

    Only destination IPs with more than one packet are considered.

    Args:
        real_df_1 (pd.DataFrame): Real dataset containing network packet data.
        gen_df_2 (pd.DataFrame): Generated dataset containing network packet data.

    Returns:
        float: Jensen-Shannon Divergence (JSD) in [0, 1].
    """
    import numpy as np

    def compute_avg_intervals(df):
        valid = df.groupby('dstip').filter(lambda x: len(x) > 1)
        return valid.sort_values('time').groupby('dstip')['time'].agg(lambda x: np.diff(x).mean()).values

    real_intervals = compute_avg_intervals(real_df_1)
    gen_intervals = compute_avg_intervals(gen_df_2)

    # Convert to distributions
    real_dist = real_intervals / real_intervals.sum() if real_intervals.sum() > 0 else np.zeros_like(real_intervals)
    gen_dist = gen_intervals / gen_intervals.sum() if gen_intervals.sum() > 0 else np.zeros_like(gen_intervals)

    # Sort for dstip-agnostic comparison
    real_sorted = np.sort(real_dist)[::-1]
    gen_sorted = np.sort(gen_dist)[::-1]

    return jensenshannon_wrapper(real_sorted, gen_sorted, base=2)


def flow_dstip_stateful_flowduration_topnvalue(real_df_1, gen_df_2, n=10):
    """
    Computes the average Absolute Relative Error (ARE) of the flow durations
    for the top N destination IPs between the real and generated datasets.
    IP addresses are ignored; only the sorted top-N flow durations are compared.

    A flow's duration is defined as the time difference between the first and last
    packets associated with each dstip.

    Args:
        real_df_1 (pd.DataFrame): Real dataset containing network packet data.
        gen_df_2 (pd.DataFrame): Generated dataset containing network packet data.
        n (int): Number of top destination IPs to consider.

    Returns:
        float: Average ARE over the top-N flow durations, capped at 1.0.
    """
    import numpy as np

    def compute_durations(df):
        grouped = df.groupby('dstip')['time']
        return (grouped.max() - grouped.min()).values

    real_durations = compute_durations(real_df_1)
    gen_durations = compute_durations(gen_df_2)

    # Take top-N durations (sorted in descending order)
    real_top = np.sort(real_durations)[-n:][::-1]
    gen_top = np.sort(gen_durations)[-n:][::-1]

    # Pad arrays to equal length
    max_len = max(len(real_top), len(gen_top))
    real_top = np.pad(real_top, (0, max_len - len(real_top)))
    gen_top = np.pad(gen_top, (0, max_len - len(gen_top)))

    # Compute relative errors
    relative_errors = np.abs(real_top - gen_top) / np.maximum(real_top, 1)
    relative_errors = np.minimum(relative_errors, 1.0)

    return min(relative_errors.mean(), 1.0)


def flow_dstip_stateful_flowduration_distribution(real_df_1, gen_df_2, n=10):
    """
    Computes the Jensen-Shannon Divergence (JSD) between the distributions
    of flow durations for each destination IP in the real and generated datasets.
    IP addresses are ignored; only the sorted distributions are compared.

    A flow's duration is defined as the time difference between the first and last
    packets associated with each dstip.

    Args:
        real_df_1 (pd.DataFrame): Real dataset containing network packet data.
        gen_df_2 (pd.DataFrame): Generated dataset containing network packet data.

    Returns:
        float: Jensen-Shannon Divergence (JSD) in [0, 1].
    """
    import numpy as np

    def compute_durations(df):
        grouped = df.groupby('dstip')['time']
        return (grouped.max() - grouped.min()).values

    real_durations = compute_durations(real_df_1)
    gen_durations = compute_durations(gen_df_2)

    # Convert to distributions
    real_dist = real_durations / real_durations.sum() if real_durations.sum() > 0 else np.zeros_like(real_durations)
    gen_dist = gen_durations / gen_durations.sum() if gen_durations.sum() > 0 else np.zeros_like(gen_durations)

    # Sort distributions (IP-agnostic)
    real_sorted = np.sort(real_dist)[::-1]
    gen_sorted = np.sort(gen_dist)[::-1]

    return jensenshannon_wrapper(real_sorted, gen_sorted, base=2)


def flow_dstip_stateful_byterate_topnvalue(real_df_1, gen_df_2, n=10):
    """
    Computes the average Absolute Relative Error (ARE) of byte rates (bytes per unit time)
    for the top N destination IPs in the real and generated datasets. The actual IP values are ignored;
    only the sorted top-N byte rates are compared.

    Args:
        real_df_1 (pd.DataFrame): Real dataset containing network packet data.
        gen_df_2 (pd.DataFrame): Generated dataset containing network packet data.
        n (int): Number of top destination IPs to consider.

    Returns:
        float: Average ARE over the top-N byte rates, capped at 1.0.
    """
    import numpy as np

    def compute_byterates(df):
        grouped = df.groupby('dstip')
        valid = grouped.filter(lambda x: len(x) > 1).groupby('dstip')
        bytes_total = valid['pkt_len'].sum()
        durations = valid['time'].agg(lambda x: x.max() - x.min())
        durations[durations == 0] = 1  # avoid division by zero
        return (bytes_total / durations)

    real_rates = compute_byterates(real_df_1).nlargest(n).values
    gen_rates = compute_byterates(gen_df_2).nlargest(n).values

    # Pad arrays to equal length
    max_len = max(len(real_rates), len(gen_rates))
    real_rates = np.pad(real_rates, (0, max_len - len(real_rates)))
    gen_rates = np.pad(gen_rates, (0, max_len - len(gen_rates)))

    # Compute relative errors
    relative_errors = np.abs(real_rates - gen_rates) / np.maximum(real_rates, 1)
    relative_errors = np.minimum(relative_errors, 1.0)

    return min(relative_errors.mean(), 1.0)


def flow_dstip_stateful_byterate_distribution(real_df_1, gen_df_2, n=10):
    """
    Computes the Jensen-Shannon Divergence (JSD) between the distributions
    of byte rates (bytes per unit time) for each destination IP in the real and generated datasets.
    IP addresses are ignored; only the sorted distributions are compared.

    Args:
        real_df_1 (pd.DataFrame): Real dataset containing network packet data.
        gen_df_2 (pd.DataFrame): Generated dataset containing network packet data.

    Returns:
        float: Jensen-Shannon Divergence (JSD) in [0, 1].
    """
    import numpy as np

    def compute_byterates(df):
        grouped = df.groupby('dstip')
        valid = grouped.filter(lambda x: len(x) > 1).groupby('dstip')
        bytes_total = valid['pkt_len'].sum()
        durations = valid['time'].agg(lambda x: x.max() - x.min())
        durations[durations == 0] = 1  # avoid division by zero
        return (bytes_total / durations).values

    real_rates = compute_byterates(real_df_1)
    gen_rates = compute_byterates(gen_df_2)

    # Convert to probability distributions
    real_dist = real_rates / real_rates.sum() if real_rates.sum() > 0 else np.zeros_like(real_rates)
    gen_dist = gen_rates / gen_rates.sum() if gen_rates.sum() > 0 else np.zeros_like(gen_rates)

    # Sort for flow-agnostic comparison
    real_sorted = np.sort(real_dist)[::-1]
    gen_sorted = np.sort(gen_dist)[::-1]

    return jensenshannon_wrapper(real_sorted, gen_sorted, base=2)


def flow_ippair_stateless_packet_topnvalue(real_df_1, gen_df_2, n=10):
    """
    Computes the average Absolute Relative Error (ARE) of the number of packets transmitted
    by the top N source-destination IP pairs (srcip, dstip) between the real and generated datasets.

    The actual IP addresses are ignored; only the sorted top-N packet counts are compared.

    Args:
        real_df_1 (pd.DataFrame): Real dataset containing network packet data.
        gen_df_2 (pd.DataFrame): Generated dataset containing network packet data.
        n (int): Number of top IP pairs to consider.

    Returns:
        float: Average ARE over the top-N IP pairs, capped at 1.0.
    """
    import numpy as np

    # Count packets per (srcip, dstip) pair
    real_counts = real_df_1.groupby(['srcip', 'dstip']).size().nlargest(n).values
    gen_counts = gen_df_2.groupby(['srcip', 'dstip']).size().nlargest(n).values

    # Pad arrays to equal length
    max_len = max(len(real_counts), len(gen_counts))
    real_counts = np.pad(real_counts, (0, max_len - len(real_counts)))
    gen_counts = np.pad(gen_counts, (0, max_len - len(gen_counts)))

    # Compute relative errors
    relative_errors = np.abs(real_counts - gen_counts) / np.maximum(real_counts, 1)
    relative_errors = np.minimum(relative_errors, 1.0)

    return min(relative_errors.mean(), 1.0)


def flow_ippair_stateless_packet_distribution(real_df_1, gen_df_2, n=10):
    """
    Computes the Jensen-Shannon Divergence (JSD) between the distribution of number of packets
    sent by each source-destination IP pair (srcip, dstip) in the real and generated datasets.
    The IP addresses are ignored; only the sorted distributions are compared.

    Args:
        real_df_1 (pd.DataFrame): Real dataset.
        gen_df_2 (pd.DataFrame): Generated dataset.

    Returns:
        float: JSD distance between the IP-agnostic packet count distributions.
    """
    from scipy.spatial.distance import jensenshannon
    import numpy as np

    # Count number of packets per (srcip, dstip) pair
    real_counts = real_df_1.groupby(['srcip', 'dstip']).size().values
    gen_counts = gen_df_2.groupby(['srcip', 'dstip']).size().values

    # Normalize to probability distributions
    real_dist = real_counts / real_counts.sum() if real_counts.sum() > 0 else np.zeros_like(real_counts)
    gen_dist = gen_counts / gen_counts.sum() if gen_counts.sum() > 0 else np.zeros_like(gen_counts)

    # Sort distributions in descending order for IP-agnostic comparison
    real_sorted = np.sort(real_dist)[::-1]
    gen_sorted = np.sort(gen_dist)[::-1]

    # Pad to equal length
    max_len = max(len(real_sorted), len(gen_sorted))
    real_padded = np.pad(real_sorted, (0, max_len - len(real_sorted)))
    gen_padded = np.pad(gen_sorted, (0, max_len - len(gen_sorted)))

    return jensenshannon_wrapper(real_padded, gen_padded, base=2)


def flow_ippair_stateless_bytes_topnvalue(real_df_1, gen_df_2, n=10):
    """
    Computes the average Absolute Relative Error (ARE) of the number of bytes transmitted
    by the top N source-destination IP pairs (srcip, dstip) between the real and generated datasets.

    The actual IP addresses are ignored; only the sorted top-N byte counts are compared.

    Args:
        real_df_1 (pd.DataFrame): Real dataset containing network packet data.
        gen_df_2 (pd.DataFrame): Generated dataset containing network packet data.
        n (int): Number of top IP pairs to consider.

    Returns:
        float: Average ARE over the top-N IP pairs, capped at 1.0.
    """
    import numpy as np

    # Sum bytes per (srcip, dstip) pair
    real_bytes = real_df_1.groupby(['srcip', 'dstip'])['pkt_len'].sum().nlargest(n).values
    gen_bytes = gen_df_2.groupby(['srcip', 'dstip'])['pkt_len'].sum().nlargest(n).values

    # Pad arrays to equal length
    max_len = max(len(real_bytes), len(gen_bytes))
    real_bytes = np.pad(real_bytes, (0, max_len - len(real_bytes)))
    gen_bytes = np.pad(gen_bytes, (0, max_len - len(gen_bytes)))

    # Compute relative errors
    relative_errors = np.abs(real_bytes - gen_bytes) / np.maximum(real_bytes, 1)
    relative_errors = np.minimum(relative_errors, 1.0)

    return min(relative_errors.mean(), 1.0)

def flow_ippair_stateless_bytes_distribution(real_df_1, gen_df_2, n=10):
    """
    Computes the Jensen-Shannon Divergence (JSD) between the byte distributions
    of source-destination IP pairs (srcip, dstip) in the real and generated datasets.
    The IP addresses are ignored; only the sorted distribution values are compared (IP-agnostic).

    Args:
        real_df_1 (pd.DataFrame): Real dataset containing network packet data.
        gen_df_2 (pd.DataFrame): Generated dataset containing network packet data.

    Returns:
        float: Jensen-Shannon Divergence (JSD) value in [0, 1].
    """
    from scipy.spatial.distance import jensenshannon
    import numpy as np

    # Compute byte sums by (srcip, dstip) pair
    real_bytes = real_df_1.groupby(['srcip', 'dstip'])['pkt_len'].sum().values
    gen_bytes = gen_df_2.groupby(['srcip', 'dstip'])['pkt_len'].sum().values

    # Normalize to obtain probability distributions
    real_dist = real_bytes / real_bytes.sum() if real_bytes.sum() > 0 else np.zeros_like(real_bytes)
    gen_dist = gen_bytes / gen_bytes.sum() if gen_bytes.sum() > 0 else np.zeros_like(gen_bytes)

    # Sort distributions (IP-agnostic)
    real_sorted = np.sort(real_dist)[::-1]
    gen_sorted = np.sort(gen_dist)[::-1]

    # Pad to equal length
    max_len = max(len(real_sorted), len(gen_sorted))
    real_padded = np.pad(real_sorted, (0, max_len - len(real_sorted)))
    gen_padded = np.pad(gen_sorted, (0, max_len - len(gen_sorted)))

    return jensenshannon_wrapper(real_padded, gen_padded, base=2)


def flow_ippair_stateless_connection2srcport_topnvalue(real_df_1, gen_df_2, n=10):
    """
    Computes the average Absolute Relative Error (ARE) of the number of distinct source ports
    each of the top N (srcip, dstip) IP pairs contacted in the real and generated datasets.
    The IP addresses are ignored; only the sorted top-N counts are compared."
    Args:
        real_df_1 (pd.DataFrame): Real dataset containing network packet data.
        gen_df_2 (pd.DataFrame): Generated dataset containing network packet data.
        n (int): Number of top IP pairs to consider.

    Returns:
        float: Average ARE over the top-N IP pairs, capped at 1.0.
    """
    import numpy as np

    # Count distinct srcport per (srcip, dstip) pair
    real_conn = real_df_1.groupby(['srcip', 'dstip'])['srcport'].nunique()
    gen_conn = gen_df_2.groupby(['srcip', 'dstip'])['srcport'].nunique()

    # Take the top N values by real count
    real_top = real_conn.sort_values(ascending=False).head(n).values
    gen_top = gen_conn.sort_values(ascending=False).head(n).values

    # Pad with zeros to equal length
    max_len = max(len(real_top), len(gen_top))
    real_top = np.pad(real_top, (0, max_len - len(real_top)))
    gen_top = np.pad(gen_top, (0, max_len - len(gen_top)))

    # Compute relative errors
    relative_errors = np.abs(real_top - gen_top) / np.maximum(real_top, 1)
    relative_errors = np.minimum(relative_errors, 1.0)

    return min(relative_errors.mean(), 1.0)


def flow_ippair_stateless_connection2srcport_distribution(real_df_1, gen_df_2, n=10):
    """
    Computes the Jensen-Shannon Divergence (JSD) between the distribution of the number
    of distinct source ports contacted by each (srcip, dstip) pair in the real and generated datasets.
    The IP addresses are ignored; only the sorted distributions are compared.

    Args:
        real_df_1 (pd.DataFrame): Real dataset containing network packet data.
        gen_df_2 (pd.DataFrame): Generated dataset containing network packet data.

    Returns:
        float: Jensen-Shannon Divergence (JSD) value in [0, 1].
    """
    from scipy.spatial.distance import jensenshannon
    import numpy as np

    # Count distinct srcports per (srcip, dstip) pair
    real_conn = real_df_1.groupby(['srcip', 'dstip'])['srcport'].nunique().values
    gen_conn = gen_df_2.groupby(['srcip', 'dstip'])['srcport'].nunique().values

    # Normalize to obtain distributions
    real_dist = real_conn / real_conn.sum() if real_conn.sum() > 0 else np.zeros_like(real_conn)
    gen_dist = gen_conn / gen_conn.sum() if gen_conn.sum() > 0 else np.zeros_like(gen_conn)

    # Sort for IP-agnostic comparison
    real_sorted = np.sort(real_dist)[::-1]
    gen_sorted = np.sort(gen_dist)[::-1]

    # Pad to equal length
    max_len = max(len(real_sorted), len(gen_sorted))
    real_padded = np.pad(real_sorted, (0, max_len - len(real_sorted)))
    gen_padded = np.pad(gen_sorted, (0, max_len - len(gen_sorted)))

    return jensenshannon_wrapper(real_padded, gen_padded, base=2)


def flow_ippair_stateless_connection2dstport_topnvalue(real_df_1, gen_df_2, n=10):
    """
    Computes the average Absolute Relative Error (ARE) of the number of distinct destination ports
    each of the top N (srcip, dstip) IP pairs contacted in the real and generated datasets.
    The IP addresses are ignored; only the sorted top-N counts are compared.

    Args:
        real_df_1 (pd.DataFrame): Real dataset containing network packet data.
        gen_df_2 (pd.DataFrame): Generated dataset containing network packet data.
        n (int): Number of top IP pairs to consider.

    Returns:
        float: Average ARE over the top-N IP pairs, capped at 1.0.
    """
    import numpy as np

    # Count distinct dstport per (srcip, dstip) pair
    real_conn = real_df_1.groupby(['srcip', 'dstip'])['dstport'].nunique()
    gen_conn = gen_df_2.groupby(['srcip', 'dstip'])['dstport'].nunique()

    # Take the top N values by real count
    real_top = real_conn.sort_values(ascending=False).head(n).values
    gen_top = gen_conn.sort_values(ascending=False).head(n).values

    # Pad with zeros to equal length
    max_len = max(len(real_top), len(gen_top))
    real_top = np.pad(real_top, (0, max_len - len(real_top)))
    gen_top = np.pad(gen_top, (0, max_len - len(gen_top)))

    # Compute relative errors
    relative_errors = np.abs(real_top - gen_top) / np.maximum(real_top, 1)
    relative_errors = np.minimum(relative_errors, 1.0)

    return min(relative_errors.mean(), 1.0)


def flow_ippair_stateless_connection2dstport_distribution(real_df_1, gen_df_2, n=10):
    """
    Computes the Jensen-Shannon Divergence (JSD) between the distribution of the number
    of distinct destination ports contacted by each (srcip, dstip) pair in the real and generated datasets.
    The IP addresses are ignored; only the sorted distributions are compared.
 
    Args:
        real_df_1 (pd.DataFrame): Real dataset containing network packet data.
        gen_df_2 (pd.DataFrame): Generated dataset containing network packet data.
 
    Returns:
        float: Jensen-Shannon Divergence (JSD) value in [0, 1].
    """
    from scipy.spatial.distance import jensenshannon
    import numpy as np
 
    # Count distinct dstport per (srcip, dstip) pair
    real_conn = real_df_1.groupby(['srcip', 'dstip'])['dstport'].nunique().values
    gen_conn = gen_df_2.groupby(['srcip', 'dstip'])['dstport'].nunique().values
 
    # Normalize to obtain distributions
    real_dist = real_conn / real_conn.sum() if real_conn.sum() > 0 else np.zeros_like(real_conn)
    gen_dist = gen_conn / gen_conn.sum() if gen_conn.sum() > 0 else np.zeros_like(gen_conn)
 
    # Sort for IP-agnostic comparison
    real_sorted = np.sort(real_dist)[::-1]
    gen_sorted = np.sort(gen_dist)[::-1]
 
    # Pad to equal length
    max_len = max(len(real_sorted), len(gen_sorted))
    real_padded = np.pad(real_sorted, (0, max_len - len(real_sorted)))
    gen_padded = np.pad(gen_sorted, (0, max_len - len(gen_sorted)))
 
    return jensenshannon_wrapper(real_padded, gen_padded, base=2)


def flow_ippair_stateless_connection2flow_topnvalue(real_df_1, gen_df_2, n=10):
    """
    Computes the average Absolute Relative Error (ARE) of the number of distinct 5-tuple flows
    contacted by each of the top N (srcip, dstip) IP pairs in the real and generated datasets.

    A connection (flow) is defined as a unique 5-tuple: (srcip, dstip, srcport, dstport, proto).
    The IP addresses are ignored; only the sorted top-N connection counts are compared.

    Args:
        real_df_1 (pd.DataFrame): Real dataset containing network packet data.
        gen_df_2 (pd.DataFrame): Generated dataset containing network packet data.
        n (int): Number of top IP pairs to consider.

    Returns:
        float: Average ARE over the top-N IP pairs, capped at 1.0.
    """
    import numpy as np

    def count_flows(df):
        df = df.copy()
        # Create a full 5-tuple identifier for each packet
        df['flow'] = list(zip(df['srcip'], df['dstip'], df['srcport'], df['dstport'], df['proto']))
        # Count distinct flows per (srcip, dstip)
        return df.groupby(['srcip', 'dstip'])['flow'].nunique()

    # Count flows for real and generated datasets
    real_counts = count_flows(real_df_1)
    gen_counts = count_flows(gen_df_2)

    # Take top-N values by real count
    real_top = real_counts.sort_values(ascending=False).head(n).values
    gen_top = gen_counts.sort_values(ascending=False).head(n).values

    # Pad arrays to the same length
    max_len = max(len(real_top), len(gen_top))
    real_top = np.pad(real_top, (0, max_len - len(real_top)))
    gen_top = np.pad(gen_top, (0, max_len - len(gen_top)))

    # Compute relative errors
    relative_errors = np.abs(real_top - gen_top) / np.maximum(real_top, 1)
    relative_errors = np.minimum(relative_errors, 1.0)

    return min(relative_errors.mean(), 1.0)


def flow_ippair_stateless_connection2flow_distribution(real_df_1, gen_df_2, n=10):
    """
    Computes the Jensen-Shannon Divergence (JSD) between the distribution of the number
    of distinct flows (defined by full 5-tuples) contacted by each (srcip, dstip) pair
    in the real and generated datasets. IP addresses are ignored; only the sorted distributions are compared.

    Args:
        real_df_1 (pd.DataFrame): Real dataset containing network packet data.
        gen_df_2 (pd.DataFrame): Generated dataset containing network packet data.

    Returns:
        float: Jensen-Shannon Divergence (JSD) value in [0, 1].
    """
    from scipy.spatial.distance import jensenshannon
    import numpy as np

    def count_flows(df):
        df = df.copy()
        df['flow'] = list(zip(df['srcip'], df['dstip'], df['srcport'], df['dstport'], df['proto']))
        return df.groupby(['srcip', 'dstip'])['flow'].nunique().values

    real_conn = count_flows(real_df_1)
    gen_conn = count_flows(gen_df_2)

    real_dist = real_conn / real_conn.sum() if real_conn.sum() > 0 else np.zeros_like(real_conn)
    gen_dist = gen_conn / gen_conn.sum() if gen_conn.sum() > 0 else np.zeros_like(gen_conn)

    real_sorted = np.sort(real_dist)[::-1]
    gen_sorted = np.sort(gen_dist)[::-1]

    max_len = max(len(real_sorted), len(gen_sorted))
    real_padded = np.pad(real_sorted, (0, max_len - len(real_sorted)))
    gen_padded = np.pad(gen_sorted, (0, max_len - len(gen_sorted)))

    return jensenshannon_wrapper(real_padded, gen_padded, base=2)


def flow_ippair_stateful_avgpacketinterval_topnvalue(real_df_1, gen_df_2, n=10):
    """
    Computes the average Absolute Relative Error (ARE) of the average packet interval
    for the top N (srcip, dstip) IP pairs between the real and generated datasets.
    IP addresses are ignored; only the sorted average intervals are compared.

    Args:
        real_df_1 (pd.DataFrame): Real dataset containing network packet data.
        gen_df_2 (pd.DataFrame): Generated dataset containing network packet data.
        n (int): Number of top (srcip, dstip) IP pairs to consider.

    Returns:
        float: Average ARE over the top-N average intervals, capped at 1.0.
    """
    import numpy as np

    def compute_avg_intervals(df):
        # Filter to only include flows with more than one packet
        valid = df.groupby(['srcip', 'dstip']).filter(lambda x: len(x) > 1)
        # Compute average packet interval for each (srcip, dstip) pair
        return valid.sort_values('time').groupby(['srcip', 'dstip'])['time'].agg(lambda x: np.diff(x).mean())

    # Compute average packet intervals
    real_intervals = compute_avg_intervals(real_df_1).nlargest(n).values
    gen_intervals = compute_avg_intervals(gen_df_2).nlargest(n).values

    # Pad both arrays to equal length
    max_len = max(len(real_intervals), len(gen_intervals))
    real_intervals = np.pad(real_intervals, (0, max_len - len(real_intervals)))
    gen_intervals = np.pad(gen_intervals, (0, max_len - len(gen_intervals)))

    # Compute relative errors
    relative_errors = np.abs(real_intervals - gen_intervals) / np.maximum(real_intervals, 1)
    relative_errors = np.minimum(relative_errors, 1.0)

    return min(relative_errors.mean(), 1.0)


def flow_ippair_stateful_avgpacketinterval_distribution(real_df_1, gen_df_2, n=10):
    """
    Computes the Jensen-Shannon Divergence (JSD) between the distributions
    of average packet intervals for each (srcip, dstip) pair in the real and generated datasets.
    IP addresses are ignored; only the sorted distributions are compared.

    Args:
        real_df_1 (pd.DataFrame): Real dataset containing network packet data.
        gen_df_2 (pd.DataFrame): Generated dataset containing network packet data.

    Returns:
        float: Jensen-Shannon Divergence (JSD) in [0, 1].
    """
    import numpy as np

    def compute_avg_intervals(df):
        valid = df.groupby(['srcip', 'dstip']).filter(lambda x: len(x) > 1)
        return valid.sort_values('time').groupby(['srcip', 'dstip'])['time'].agg(lambda x: np.diff(x).mean()).values

    real_intervals = compute_avg_intervals(real_df_1)
    gen_intervals = compute_avg_intervals(gen_df_2)

    # Convert to probability distributions
    real_dist = real_intervals / real_intervals.sum() if real_intervals.sum() > 0 else np.zeros_like(real_intervals)
    gen_dist = gen_intervals / gen_intervals.sum() if gen_intervals.sum() > 0 else np.zeros_like(gen_intervals)

    # Sort for IP-pair-agnostic comparison
    real_sorted = np.sort(real_dist)[::-1]
    gen_sorted = np.sort(gen_dist)[::-1]

    return jensenshannon_wrapper(real_sorted, gen_sorted, base=2)


def flow_ippair_stateful_flowduration_topnvalue(real_df_1, gen_df_2, n=10):
    """
    Computes the average Absolute Relative Error (ARE) of the flow durations
    for the top N source-destination IP pairs (srcip, dstip) between the real and generated datasets.
    IP addresses are ignored; only the sorted top-N flow durations are compared.

    A flow's duration is defined as the time difference between the first and last
    packets associated with each (srcip, dstip) pair.

    Args:
        real_df_1 (pd.DataFrame): Real dataset containing network packet data.
        gen_df_2 (pd.DataFrame): Generated dataset containing network packet data.
        n (int): Number of top (srcip, dstip) pairs to consider.

    Returns:
        float: Average ARE over the top-N flow durations, capped at 1.0.
    """
    import numpy as np

    def compute_durations(df):
        grouped = df.groupby(['srcip', 'dstip'])['time']
        return (grouped.max() - grouped.min()).values

    real_durations = compute_durations(real_df_1)
    gen_durations = compute_durations(gen_df_2)

    # Take top-N durations (sorted in descending order)
    real_top = np.sort(real_durations)[-n:][::-1]
    gen_top = np.sort(gen_durations)[-n:][::-1]

    # Pad arrays to equal length
    max_len = max(len(real_top), len(gen_top))
    real_top = np.pad(real_top, (0, max_len - len(real_top)))
    gen_top = np.pad(gen_top, (0, max_len - len(gen_top)))

    # Compute relative errors
    relative_errors = np.abs(real_top - gen_top) / np.maximum(real_top, 1)
    relative_errors = np.minimum(relative_errors, 1.0)

    return min(relative_errors.mean(), 1.0)


def flow_ippair_stateful_flowduration_distribution(real_df_1, gen_df_2, n=10):
    """
    Computes the Jensen-Shannon Divergence (JSD) between the distributions
    of flow durations for each (srcip, dstip) pair in the real and generated datasets.
    IP addresses are ignored; only the sorted distributions are compared.

    A flow's duration is defined as the time difference between the first and last
    packets associated with each (srcip, dstip) pair.

    Args:
        real_df_1 (pd.DataFrame): Real dataset containing network packet data.
        gen_df_2 (pd.DataFrame): Generated dataset containing network packet data.

    Returns:
        float: Jensen-Shannon Divergence (JSD) in [0, 1].
    """
    import numpy as np

    def compute_durations(df):
        grouped = df.groupby(['srcip', 'dstip'])['time']
        return (grouped.max() - grouped.min()).values

    real_durations = compute_durations(real_df_1)
    gen_durations = compute_durations(gen_df_2)

    # Convert to distributions (sums may be large)
    real_dist = real_durations / real_durations.sum() if real_durations.sum() > 0 else np.zeros_like(real_durations)
    gen_dist = gen_durations / gen_durations.sum() if gen_durations.sum() > 0 else np.zeros_like(gen_durations)

    # Sort for flow-agnostic comparison
    real_sorted = np.sort(real_dist)[::-1]
    gen_sorted = np.sort(gen_dist)[::-1]

    return jensenshannon_wrapper(real_sorted, gen_sorted, base=2)


def flow_ippair_stateful_byterate_topnvalue(real_df_1, gen_df_2, n=10):
    """
    Computes the average Absolute Relative Error (ARE) of byte rates (bytes per unit time)
    for the top N source-destination IP pairs (srcip, dstip) in the real and generated datasets.
    IP addresses are ignored; only the sorted top-N byte rates are compared.

    Args:
        real_df_1 (pd.DataFrame): Real dataset containing network packet data.
        gen_df_2 (pd.DataFrame): Generated dataset containing network packet data.
        n (int): Number of top (srcip, dstip) IP pairs to consider.

    Returns:
        float: Average ARE over the top-N byte rates, capped at 1.0.
    """
    import numpy as np

    def compute_byterates(df):
        grouped = df.groupby(['srcip', 'dstip'])
        # Filter out flows with only one packet
        valid = grouped.filter(lambda x: len(x) > 1).groupby(['srcip', 'dstip'])
        bytes_total = valid['pkt_len'].sum()
        durations = valid['time'].agg(lambda x: x.max() - x.min())
        durations[durations == 0] = 1  # prevent division by zero
        return bytes_total / durations

    real_rates = compute_byterates(real_df_1).nlargest(n).values
    gen_rates = compute_byterates(gen_df_2).nlargest(n).values

    # Pad arrays to equal length
    max_len = max(len(real_rates), len(gen_rates))
    real_rates = np.pad(real_rates, (0, max_len - len(real_rates)))
    gen_rates = np.pad(gen_rates, (0, max_len - len(gen_rates)))

    # Compute relative errors
    relative_errors = np.abs(real_rates - gen_rates) / np.maximum(real_rates, 1)
    relative_errors = np.minimum(relative_errors, 1.0)

    return min(relative_errors.mean(), 1.0)


def flow_ippair_stateful_byterate_distribution(real_df_1, gen_df_2, n=10):
    """
    Computes the Jensen-Shannon Divergence (JSD) between the byte rate distributions
    of each (srcip, dstip) pair in the real and generated datasets.
    Flow identities are ignored; only the sorted distributions are compared.

    Byte rate is defined as total bytes divided by flow duration. Only flows
    with more than one packet are considered.

    Args:
        real_df_1 (pd.DataFrame): Real dataset containing network packet data.
        gen_df_2 (pd.DataFrame): Generated dataset containing network packet data.

    Returns:
        float: Jensen-Shannon Divergence (JSD) in [0, 1].
    """
    import numpy as np

    def compute_byterates(df):
        grouped = df.groupby(['srcip', 'dstip'])
        valid = grouped.filter(lambda x: len(x) > 1).groupby(['srcip', 'dstip'])
        bytes_total = valid['pkt_len'].sum()
        durations = valid['time'].agg(lambda x: x.max() - x.min())
        durations[durations == 0] = 1  # avoid division by zero
        return (bytes_total / durations).values

    real_rates = compute_byterates(real_df_1)
    gen_rates = compute_byterates(gen_df_2)

    # Convert to probability distributions
    real_dist = real_rates / real_rates.sum() if real_rates.sum() > 0 else np.zeros_like(real_rates)
    gen_dist = gen_rates / gen_rates.sum() if gen_rates.sum() > 0 else np.zeros_like(gen_rates)

    # Sort for flow-agnostic comparison
    real_sorted = np.sort(real_dist)[::-1]
    gen_sorted = np.sort(gen_dist)[::-1]

    return jensenshannon_wrapper(real_sorted, gen_sorted, base=2)


def flow_srcport_stateless_packet_topnkey(real_df_1, gen_df_2, n=10):
    """
    Computes the 1-hit rate between the top-N source ports (by packet count) in the real and generated datasets.
    This measures how many of the real top-N ports are present in the generated top-N ports.

    Args:
        real_df_1 (pd.DataFrame): Real dataset containing network packet data.
        gen_df_2 (pd.DataFrame): Generated dataset containing network packet data.
        n (int): Number of top source ports to consider.

    Returns:
        float: 1-hit rate (recall@N) between real and generated top-N source ports.
    """
    # Get top-N source ports by count in both datasets
    real_top_ports = set(real_df_1['srcport'].value_counts().nlargest(n).index)
    gen_top_ports = set(gen_df_2['srcport'].value_counts().nlargest(n).index)

    # Compute how many of the real top ports are in the generated top ports
    hits = len(real_top_ports.intersection(gen_top_ports))

    return 1 - hits / n


def flow_srcport_stateless_packet_topnvalue(real_df_1, gen_df_2, n=10):
    """
    Computes the average Absolute Relative Error (ARE) of the number of packets sent
    by the top N source ports between the real and generated datasets.
    The actual port values are used; comparison is not port-agnostic.
 
    Args:
        real_df_1 (pd.DataFrame): Real dataset containing network packet data.
        gen_df_2 (pd.DataFrame): Generated dataset containing network packet data.
        n (int): Number of top source ports to consider.
 
    Returns:
        float: Average ARE over the top-N source ports, capped at 1.0.
    """
    import numpy as np
 
    # Get top-N packet counts from real data (port-agnostic)
    real_counts = real_df_1['srcport'].value_counts().nlargest(n).values
 
    # Get top-N packet counts from generated data (port-agnostic)
    gen_counts = gen_df_2['srcport'].value_counts().nlargest(n).values
 
    # Pad to the same length
    max_len = max(len(real_counts), len(gen_counts))
    real_counts = np.pad(real_counts, (0, max_len - len(real_counts)))
    gen_counts = np.pad(gen_counts, (0, max_len - len(gen_counts)))
 
    # Compute relative errors
    relative_errors = np.abs(real_counts - gen_counts) / np.maximum(real_counts, 1)
    relative_errors = np.minimum(relative_errors, 1.0)
 
    return min(relative_errors.mean(), 1.0)


def flow_srcport_stateless_bytes_topnkey(real_df_1, gen_df_2, n=10):
    """
    Computes the 1-hit rate between the top-N source ports (by total bytes sent)
    in the real and generated datasets. This measures how many of the real top-N ports
    are also present in the generated top-N ports.
 
    Args:
        real_df_1 (pd.DataFrame): Real dataset containing network packet data.
        gen_df_2 (pd.DataFrame): Generated dataset containing network packet data.
        n (int): Number of top source ports to consider.
 
    Returns:
        float: 1-hit rate (recall@N) between real and generated top-N source ports by byte volume.
    """
    # Get top-N source ports by byte volume
    real_top_ports = set(real_df_1.groupby('srcport')['pkt_len'].sum().nlargest(n).index)
    gen_top_ports = set(gen_df_2.groupby('srcport')['pkt_len'].sum().nlargest(n).index)
 
    # Count how many of the real top ports are in the generated top ports
    hits = len(real_top_ports.intersection(gen_top_ports))
 
    return 1 - hits / n


def flow_srcport_stateless_bytes_topnvalue(real_df_1, gen_df_2, n=10):
    """
    Computes the average Absolute Relative Error (ARE) of the number of bytes sent
    by the top N source ports between the real and generated datasets.
    The actual port values are used; comparison is not port-agnostic.
    
    Args:
        real_df_1 (pd.DataFrame): Real dataset containing network packet data.
        gen_df_2 (pd.DataFrame): Generated dataset containing network packet data.
        n (int): Number of top source ports to consider.
    
    Returns:
        float: Average ARE over the top-N source ports, capped at 1.0.
    """
    import numpy as np

    # Compute top-N byte counts from each dataset in sorted order
    real_bytes = real_df_1.groupby('srcport')['pkt_len'].sum().nlargest(n).values
    gen_bytes = gen_df_2.groupby('srcport')['pkt_len'].sum().nlargest(n).values

    max_len = max(len(real_bytes), len(gen_bytes))
    real_bytes = np.pad(real_bytes, (0, max_len - len(real_bytes)))
    gen_bytes = np.pad(gen_bytes, (0, max_len - len(gen_bytes)))

    relative_errors = np.abs(real_bytes - gen_bytes) / np.maximum(real_bytes, 1)
    relative_errors = np.minimum(relative_errors, 1.0)

    return min(relative_errors.mean(), 1.0)


def flow_srcport_stateless_bytes_distribution(real_df_1, gen_df_2, n=10):
    """
    Computes the Jensen-Shannon Divergence (JSD) between the byte distributions
    of source ports in the real and generated datasets. This comparison is port-specific (not agnostic).

    Args:
        real_df_1 (pd.DataFrame): Real dataset containing network packet data.
        gen_df_2 (pd.DataFrame): Generated dataset containing network packet data.

    Returns:
        float: Jensen-Shannon Divergence (JSD) value in [0, 1].
    """
    from scipy.spatial.distance import jensenshannon
    import numpy as np

    # Compute total bytes sent from each source port
    real_bytes = real_df_1.groupby('srcport')['pkt_len'].sum()
    gen_bytes = gen_df_2.groupby('srcport')['pkt_len'].sum()

    # Normalize to obtain probability distributions
    real_dist = real_bytes / real_bytes.sum() if real_bytes.sum() > 0 else np.zeros_like(real_bytes)
    gen_dist = gen_bytes / gen_bytes.sum() if gen_bytes.sum() > 0 else np.zeros_like(gen_bytes)

    # Align distributions by source port
    all_ports = set(real_dist.index).union(set(gen_dist.index))
    real_aligned = real_dist.reindex(all_ports, fill_value=0).sort_index()
    gen_aligned = gen_dist.reindex(all_ports, fill_value=0).sort_index()

    return jensenshannon_wrapper(real_aligned.values, gen_aligned.values, base=2)


def flow_srcport_stateless_connection2srcip_topnkey(real_df_1, gen_df_2, n=10):
    """
    Computes the 1-hit rate between the top-N source ports (by number of distinct source IPs)
    in the real and generated datasets. This measures how many of the real top-N ports are also
    present in the generated top-N ports. Lower values indicate greater distance.

    Args:
        real_df_1 (pd.DataFrame): Real dataset containing network packet data.
        gen_df_2 (pd.DataFrame): Generated dataset containing network packet data.
        n (int): Number of top source ports to consider.

    Returns:
        float: 1 - hit rate between real and generated top-N source ports.
    """
    # Identify top-N source ports by number of distinct source IPs
    real_top_ports = set(real_df_1.groupby('srcport')['srcip'].nunique().nlargest(n).index)
    gen_top_ports = set(gen_df_2.groupby('srcport')['srcip'].nunique().nlargest(n).index)

    # Count overlap between real and generated top ports
    hits = len(real_top_ports.intersection(gen_top_ports))

    return 1 - hits / n


def flow_srcport_stateless_connection2srcip_topnvalue(real_df_1, gen_df_2, n=10):
    """
    Computes the average Absolute Relative Error (ARE) of the number of distinct source IPs
    connected to the top N source ports (by distinct srcip count) in the real dataset, comparing
    the same ports' values in the generated dataset. The actual port values are used.
 
    Args:
        real_df_1 (pd.DataFrame): Real dataset containing network packet data.
        gen_df_2 (pd.DataFrame): Generated dataset containing network packet data.
        n (int): Number of top source ports to consider.
 
    Returns:
        float: Average ARE over the top-N source ports, capped at 1.0.
    """
    import numpy as np
 
    # Get number of distinct srcips per srcport
    real_conn = real_df_1.groupby('srcport')['srcip'].nunique().nlargest(n).values
    gen_conn = gen_df_2.groupby('srcport')['srcip'].nunique().nlargest(n).values
 
    # Pad to equal length
    max_len = max(len(real_conn), len(gen_conn))
    real_conn = np.pad(real_conn, (0, max_len - len(real_conn)))
    gen_conn = np.pad(gen_conn, (0, max_len - len(gen_conn)))
 
    # Compute relative errors
    relative_errors = np.abs(real_conn - gen_conn) / np.maximum(real_conn, 1)
    relative_errors = np.minimum(relative_errors, 1.0)
 
    return min(relative_errors.mean(), 1.0)


def flow_srcport_stateless_connection2srcip_distribution(real_df_1, gen_df_2, n=10):
    """
    Computes the Jensen-Shannon Divergence (JSD) between the distribution of the number
    of distinct source IPs connected to each source port in the real and generated datasets.
    The comparison is not port-agnostic; it matches counts by specific source ports.

    Args:
        real_df_1 (pd.DataFrame): Real dataset containing network packet data.
        gen_df_2 (pd.DataFrame): Generated dataset containing network packet data.

    Returns:
        float: Jensen-Shannon Divergence (JSD) value in [0, 1].
    """
    from scipy.spatial.distance import jensenshannon
    import numpy as np

    # Count number of distinct source IPs per source port
    real_conn = real_df_1.groupby('srcport')['srcip'].nunique()
    gen_conn = gen_df_2.groupby('srcport')['srcip'].nunique()

    # Normalize to probability distributions
    real_dist = real_conn / real_conn.sum() if real_conn.sum() > 0 else np.zeros_like(real_conn)
    gen_dist = gen_conn / gen_conn.sum() if gen_conn.sum() > 0 else np.zeros_like(gen_conn)

    # Align by source port
    all_ports = set(real_dist.index).union(set(gen_dist.index))
    real_aligned = real_dist.reindex(all_ports, fill_value=0).sort_index()
    gen_aligned = gen_dist.reindex(all_ports, fill_value=0).sort_index()

    return jensenshannon_wrapper(real_aligned.values, gen_aligned.values, base=2)


def flow_srcport_stateless_connection2dstip_topnkey(real_df_1, gen_df_2, n=10):
    """
    Computes the 1-hit rate between the top-N source ports (by number of distinct destination IPs)
    in the real and generated datasets. This measures how many of the real top-N ports are also
    present in the generated top-N ports. Lower values indicate greater distance.

    Args:
        real_df_1 (pd.DataFrame): Real dataset containing network packet data.
        gen_df_2 (pd.DataFrame): Generated dataset containing network packet data.
        n (int): Number of top source ports to consider.

    Returns:
        float: 1 - hit rate between real and generated top-N source ports.
    """
    # Identify top-N source ports by number of distinct destination IPs
    real_top_ports = set(real_df_1.groupby('srcport')['dstip'].nunique().nlargest(n).index)
    gen_top_ports = set(gen_df_2.groupby('srcport')['dstip'].nunique().nlargest(n).index)

    # Count overlap between real and generated top ports
    hits = len(real_top_ports.intersection(gen_top_ports))

    return 1 - hits / n


def flow_srcport_stateless_connection2dstip_topnvalue(real_df_1, gen_df_2, n=10):
    """
    Computes the average Absolute Relative Error (ARE) of the number of distinct destination IPs
    contacted by the top N source ports between the real and generated datasets.
    The actual port values are used (not port-agnostic); comparison is based on the values only.

    Args:
        real_df_1 (pd.DataFrame): Real dataset containing network packet data.
        gen_df_2 (pd.DataFrame): Generated dataset containing network packet data.
        n (int): Number of top source ports to consider.

    Returns:
        float: Average ARE over the top-N source ports, capped at 1.0.
    """
    import numpy as np

    # Count distinct destination IPs per source port
    real_conn = real_df_1.groupby('srcport')['dstip'].nunique().nlargest(n).values
    gen_conn = gen_df_2.groupby('srcport')['dstip'].nunique().nlargest(n).values

    # Pad with zeros to equal length
    max_len = max(len(real_conn), len(gen_conn))
    real_conn = np.pad(real_conn, (0, max_len - len(real_conn)))
    gen_conn = np.pad(gen_conn, (0, max_len - len(gen_conn)))

    # Compute relative errors
    relative_errors = np.abs(real_conn - gen_conn) / np.maximum(real_conn, 1)
    relative_errors = np.minimum(relative_errors, 1.0)

    return min(relative_errors.mean(), 1.0)


def flow_srcport_stateless_connection2dstip_distribution(real_df_1, gen_df_2, n=10):
    """
    Computes the Jensen-Shannon Divergence (JSD) between the distribution of the number
    of distinct destination IPs connected to each source port in the real and generated datasets.
    The comparison is not port-agnostic; it matches counts by specific source ports.

    Args:
        real_df_1 (pd.DataFrame): Real dataset containing network packet data.
        gen_df_2 (pd.DataFrame): Generated dataset containing network packet data.

    Returns:
        float: Jensen-Shannon Divergence (JSD) value in [0, 1].
    """
    from scipy.spatial.distance import jensenshannon
    import numpy as np

    # Count number of distinct destination IPs per source port
    real_conn = real_df_1.groupby('srcport')['dstip'].nunique()
    gen_conn = gen_df_2.groupby('srcport')['dstip'].nunique()

    # Normalize to probability distributions
    real_dist = real_conn / real_conn.sum() if real_conn.sum() > 0 else np.zeros_like(real_conn)
    gen_dist = gen_conn / gen_conn.sum() if gen_conn.sum() > 0 else np.zeros_like(gen_conn)

    # Align both distributions by source port
    all_ports = set(real_dist.index).union(set(gen_dist.index))
    real_aligned = real_dist.reindex(all_ports, fill_value=0).sort_index()
    gen_aligned = gen_dist.reindex(all_ports, fill_value=0).sort_index()

    # Compute and return Jensen-Shannon Divergence
    return jensenshannon_wrapper(real_aligned.values, gen_aligned.values, base=2)


def flow_srcport_stateless_connection2dstport_topnkey(real_df_1, gen_df_2, n=10):
    """
    Computes the 1-hit rate between the top-N source ports (by number of distinct destination ports)
    in the real and generated datasets. This measures how many of the real top-N ports are also
    present in the generated top-N ports. Lower values indicate greater distance.

    Args:
        real_df_1 (pd.DataFrame): Real dataset containing network packet data.
        gen_df_2 (pd.DataFrame): Generated dataset containing network packet data.
        n (int): Number of top source ports to consider.

    Returns:
        float: 1 - hit rate between real and generated top-N source ports.
    """
    # Identify top-N source ports by number of distinct destination ports
    real_top_ports = set(real_df_1.groupby('srcport')['dstport'].nunique().nlargest(n).index)
    gen_top_ports = set(gen_df_2.groupby('srcport')['dstport'].nunique().nlargest(n).index)

    # Count how many of the real top ports are in the generated top ports
    hits = len(real_top_ports.intersection(gen_top_ports))

    return 1 - hits / n


def flow_srcport_stateless_connection2dstport_topnvalue(real_df_1, gen_df_2, n=10):
    """
    Computes the average Absolute Relative Error (ARE) of the number of distinct destination ports
    contacted by the top N source ports between the real and generated datasets.
    The actual port values are used (not port-agnostic); comparison is based on the values only.

    Args:
        real_df_1 (pd.DataFrame): Real dataset containing network packet data.
        gen_df_2 (pd.DataFrame): Generated dataset containing network packet data.
        n (int): Number of top source ports to consider.

    Returns:
        float: Average ARE over the top-N source ports, capped at 1.0.
    """
    import numpy as np

    # Count distinct destination ports per source port
    real_conn = real_df_1.groupby('srcport')['dstport'].nunique().nlargest(n).values
    gen_conn = gen_df_2.groupby('srcport')['dstport'].nunique().nlargest(n).values

    # Pad arrays to equal length
    max_len = max(len(real_conn), len(gen_conn))
    real_conn = np.pad(real_conn, (0, max_len - len(real_conn)))
    gen_conn = np.pad(gen_conn, (0, max_len - len(gen_conn)))

    # Compute relative errors
    relative_errors = np.abs(real_conn - gen_conn) / np.maximum(real_conn, 1)
    relative_errors = np.minimum(relative_errors, 1.0)

    return min(relative_errors.mean(), 1.0)


def flow_srcport_stateless_connection2dstport_distribution(real_df_1, gen_df_2, n=10):
    from scipy.spatial.distance import jensenshannon
    import numpy as np
    
    # Count number of distinct destination ports per source port
    real_conn = real_df_1.groupby('srcport')['dstport'].nunique()
    gen_conn = gen_df_2.groupby('srcport')['dstport'].nunique()
    
    # Normalize to probability distributions
    real_dist = real_conn / real_conn.sum() if real_conn.sum() > 0 else np.zeros_like(real_conn)
    gen_dist = gen_conn / gen_conn.sum() if gen_conn.sum() > 0 else np.zeros_like(gen_conn)
    
    # Align by source port
    all_ports = set(real_dist.index).union(set(gen_dist.index))
    real_aligned = real_dist.reindex(all_ports, fill_value=0).sort_index()
    gen_aligned = gen_dist.reindex(all_ports, fill_value=0).sort_index()
    
    return jensenshannon_wrapper(real_aligned.values, gen_aligned.values, base=2)


def flow_srcport_stateless_connection2dstipport_topnkey(real_df_1, gen_df_2, n=10):
    """
    Computes the 1-hit rate between the top-N source ports (by number of distinct (dstip, dstport) pairs)
    in the real and generated datasets. This measures how many of the real top-N ports are also
    present in the generated top-N ports. Lower values indicate greater distance.

    Args:
        real_df_1 (pd.DataFrame): Real dataset containing network packet data.
        gen_df_2 (pd.DataFrame): Generated dataset containing network packet data.
        n (int): Number of top source ports to consider.

    Returns:
        float: 1 - hit rate between real and generated top-N source ports.
    """
    # Count number of distinct (dstip, dstport) pairs per source port
    real_top_ports = set(
        real_df_1.groupby('srcport').apply(lambda df: df[['dstip', 'dstport']].drop_duplicates().shape[0])
        .nlargest(n).index
    )
    gen_top_ports = set(
        gen_df_2.groupby('srcport').apply(lambda df: df[['dstip', 'dstport']].drop_duplicates().shape[0])
        .nlargest(n).index
    )

    hits = len(real_top_ports.intersection(gen_top_ports))
    return 1 - hits / n


def flow_srcport_stateless_connection2dstipport_topnvalue(real_df_1, gen_df_2, n=10):
    """
    Computes the average Absolute Relative Error (ARE) of the number of distinct (dstip, dstport) pairs
    contacted by the top N source ports between the real and generated datasets.
    The port numbers themselves are ignored; only the sorted top-N counts are compared.

    Args:
        real_df_1 (pd.DataFrame): Real dataset containing network packet data.
        gen_df_2 (pd.DataFrame): Generated dataset containing network packet data.
        n (int): Number of top source ports to consider.

    Returns:
        float: Average ARE over the top-N source ports, capped at 1.0.
    """
    import numpy as np

    # Count distinct (dstip, dstport) pairs per srcport
    real_conn = real_df_1.groupby('srcport').apply(lambda df: df[['dstip', 'dstport']].drop_duplicates().shape[0])
    gen_conn = gen_df_2.groupby('srcport').apply(lambda df: df[['dstip', 'dstport']].drop_duplicates().shape[0])

    # Extract the top-N values (port-agnostic)
    real_top = real_conn.nlargest(n).values
    gen_top = gen_conn.nlargest(n).values

    # Pad arrays to equal length
    max_len = max(len(real_top), len(gen_top))
    real_top = np.pad(real_top, (0, max_len - len(real_top)))
    gen_top = np.pad(gen_top, (0, max_len - len(gen_top)))

    # Compute relative errors
    relative_errors = np.abs(real_top - gen_top) / np.maximum(real_top, 1)
    relative_errors = np.minimum(relative_errors, 1.0)

    return min(relative_errors.mean(), 1.0)


def flow_srcport_stateless_connection2dstipport_distribution(real_df_1, gen_df_2, n=10):
    """
    Computes the Jensen-Shannon Divergence (JSD) between the distribution of the number
    of distinct (dstip, dstport) pairs contacted by each source port in the real and generated datasets.
    The comparison is not port-agnostic; distributions are matched by specific source ports.

    Args:
        real_df_1 (pd.DataFrame): Real dataset containing network packet data.
        gen_df_2 (pd.DataFrame): Generated dataset containing network packet data.

    Returns:
        float: Jensen-Shannon Divergence (JSD) value in [0, 1].
    """
    from scipy.spatial.distance import jensenshannon
    import numpy as np

    # Count number of distinct (dstip, dstport) pairs per source port
    real_conn = real_df_1.groupby('srcport').apply(lambda df: df[['dstip', 'dstport']].drop_duplicates().shape[0])
    gen_conn = gen_df_2.groupby('srcport').apply(lambda df: df[['dstip', 'dstport']].drop_duplicates().shape[0])

    # Normalize to obtain probability distributions
    real_dist = real_conn / real_conn.sum() if real_conn.sum() > 0 else np.zeros_like(real_conn)
    gen_dist = gen_conn / gen_conn.sum() if gen_conn.sum() > 0 else np.zeros_like(gen_conn)

    # Align by source port
    all_ports = set(real_dist.index).union(set(gen_dist.index))
    real_aligned = real_dist.reindex(all_ports, fill_value=0).sort_index()
    gen_aligned = gen_dist.reindex(all_ports, fill_value=0).sort_index()

    return jensenshannon_wrapper(real_aligned.values, gen_aligned.values, base=2)


def flow_srcport_stateless_connection2flow_topnkey(real_df_1, gen_df_2, n=10):
    """
    Computes the 1-hit rate between the top-N source ports (by number of distinct flows)
    in the real and generated datasets. A flow is defined as a unique 5-tuple:
    (srcip, dstip, srcport, dstport, proto). This metric evaluates whether the same
    top-N source ports (by flow count) appear in both datasets.

    Args:
        real_df_1 (pd.DataFrame): Real dataset containing network packet data.
        gen_df_2 (pd.DataFrame): Generated dataset containing network packet data.
        n (int): Number of top source ports to consider.

    Returns:
        float: 1 - hit rate between the top-N source ports by flow count.
    """
    # Define flow as 5-tuple
    real_df_1 = real_df_1.copy()
    gen_df_2 = gen_df_2.copy()
    real_df_1['flow'] = list(zip(real_df_1['srcip'], real_df_1['dstip'], real_df_1['srcport'], real_df_1['dstport'], real_df_1['proto']))
    gen_df_2['flow'] = list(zip(gen_df_2['srcip'], gen_df_2['dstip'], gen_df_2['srcport'], gen_df_2['dstport'], gen_df_2['proto']))

    # Count number of distinct flows per srcport
    real_top_ports = set(real_df_1.groupby('srcport')['flow'].nunique().nlargest(n).index)
    gen_top_ports = set(gen_df_2.groupby('srcport')['flow'].nunique().nlargest(n).index)

    # Compute hit rate
    hits = len(real_top_ports.intersection(gen_top_ports))
    return 1 - hits / n


def flow_srcport_stateless_connection2flow_topnvalue(real_df_1, gen_df_2, n=10):
    """
    Computes the average Absolute Relative Error (ARE) of the number of distinct flows
    (5-tuples: srcip, dstip, srcport, dstport, proto) per source port for the top-N highest-flow source ports.

    The port values themselves are ignored; only the top-N flow counts are compared.

    Args:
        real_df_1 (pd.DataFrame): Real dataset containing network packet data.
        gen_df_2 (pd.DataFrame): Generated dataset containing network packet data.
        n (int): Number of top source ports to consider.

    Returns:
        float: Average ARE over the top-N source ports, capped at 1.0.
    """
    import numpy as np

    def count_flows(df):
        df = df.copy()
        df['flow'] = list(zip(df['srcip'], df['dstip'], df['srcport'], df['dstport'], df['proto']))
        return df.groupby('srcport')['flow'].nunique()

    # Count distinct flows per source port
    real_counts = count_flows(real_df_1)
    gen_counts = count_flows(gen_df_2)

    # Get top-N values
    real_top = real_counts.sort_values(ascending=False).head(n).values
    gen_top = gen_counts.sort_values(ascending=False).head(n).values

    # Pad to equal length
    max_len = max(len(real_top), len(gen_top))
    real_top = np.pad(real_top, (0, max_len - len(real_top)))
    gen_top = np.pad(gen_top, (0, max_len - len(gen_top)))

    # Compute relative errors
    relative_errors = np.abs(real_top - gen_top) / np.maximum(real_top, 1)
    relative_errors = np.minimum(relative_errors, 1.0)

    return min(relative_errors.mean(), 1.0)


def flow_srcport_stateless_connection2flow_distribution(real_df_1, gen_df_2, n=10):
    """
    Computes the Jensen-Shannon Divergence (JSD) between the distribution of the number
    of distinct flows (5-tuples: srcip, dstip, srcport, dstport, proto) per source port
    in the real and generated datasets. Port values are used (not agnostic).

    Args:
        real_df_1 (pd.DataFrame): Real dataset containing network packet data.
        gen_df_2 (pd.DataFrame): Generated dataset containing network packet data.

    Returns:
        float: Jensen-Shannon Divergence (JSD) value in [0, 1].
    """
    from scipy.spatial.distance import jensenshannon
    import numpy as np

    def count_flows(df):
        df = df.copy()
        df['flow'] = list(zip(df['srcip'], df['dstip'], df['srcport'], df['dstport'], df['proto']))
        return df.groupby('srcport')['flow'].nunique()

    real_conn = count_flows(real_df_1)
    gen_conn = count_flows(gen_df_2)

    real_dist = real_conn / real_conn.sum() if real_conn.sum() > 0 else np.zeros_like(real_conn)
    gen_dist = gen_conn / gen_conn.sum() if gen_conn.sum() > 0 else np.zeros_like(gen_conn)

    all_ports = set(real_dist.index).union(set(gen_dist.index))
    real_aligned = real_dist.reindex(all_ports, fill_value=0).sort_index()
    gen_aligned = gen_dist.reindex(all_ports, fill_value=0).sort_index()

    return jensenshannon_wrapper(real_aligned.values, gen_aligned.values, base=2)


def flow_srcport_stateful_avgpacketinterval_topnkey(real_df_1, gen_df_2, n=10):
    """
    Computes the 1 - hit rate between the top-N source ports (by average packet interval)
    in the real and generated datasets. A flow is defined by srcport, and only flows
    with more than one packet are considered. Lower values indicate greater similarity.

    Args:
        real_df_1 (pd.DataFrame): Real dataset containing network packet data.
        gen_df_2 (pd.DataFrame): Generated dataset containing network packet data.
        n (int): Number of top source ports to consider.

    Returns:
        float: 1 - hit rate (i.e., distance) between real and generated top-N source ports by avg packet interval.
    """
    import numpy as np

    def compute_avg_intervals(df):
        df = df.sort_values('time')
        valid = df.groupby('srcport').filter(lambda x: len(x) > 1)
        return valid.groupby('srcport')['time'].agg(lambda x: np.diff(x).mean())

    # Get top-N source ports by average interval
    real_avg_intervals = compute_avg_intervals(real_df_1).nlargest(n)
    gen_avg_intervals = compute_avg_intervals(gen_df_2).nlargest(n)

    real_top_ports = set(real_avg_intervals.index)
    gen_top_ports = set(gen_avg_intervals.index)

    hits = len(real_top_ports.intersection(gen_top_ports))
    return 1 - hits / n


def flow_srcport_stateful_avgpacketinterval_topnvalue(real_df_1, gen_df_2, n=10):
    """
    Computes the Wasserstein distance (Earth Mover's Distance) between the normalized top-N
    average packet intervals of source ports between the real and generated datasets.

    The comparison is port-agnostic; only the sorted top-N average intervals are compared.

    Args:
        real_df_1 (pd.DataFrame): Real dataset containing network packet data.
        gen_df_2 (pd.DataFrame): Generated dataset containing network packet data.
        n (int): Number of top source ports to consider.

    Returns:
        float: Wasserstein distance (p=1) over the top-N average intervals.
    """
    
    import numpy as np

    def compute_avg_intervals(df):
        df = df.sort_values('time')
        valid = df.groupby('srcport').filter(lambda x: len(x) > 1)
        return valid.groupby('srcport')['time'].agg(lambda x: np.diff(x).mean())

    real_intervals = compute_avg_intervals(real_df_1).nlargest(n).values
    gen_intervals = compute_avg_intervals(gen_df_2).nlargest(n).values

    # Compute relative errors
    relative_errors = np.abs(real_intervals - gen_intervals) / np.maximum(real_intervals, 1)
    relative_errors = np.minimum(relative_errors, 1.0)

    return min(relative_errors.mean(), 1.0)


def flow_srcport_stateful_avgpacketinterval_distribution(real_df_1, gen_df_2, n=10):
    """
    Computes the Jensen-Shannon Divergence (JSD) between the average packet interval
    distributions of each source port in the real and generated datasets.
    Port values are ignored; only the sorted distributions are compared.

    Only ports with more than one packet are considered.

    Args:
        real_df_1 (pd.DataFrame): Real dataset containing network packet data.
        gen_df_2 (pd.DataFrame): Generated dataset containing network packet data.

    Returns:
        float: Jensen-Shannon Divergence (JSD) in [0, 1].
    """
    import numpy as np

    def compute_avg_intervals(df):
        valid = df.groupby('srcport').filter(lambda x: len(x) > 1)
        return valid.sort_values('time').groupby('srcport')['time'].agg(lambda x: np.diff(x).mean()).values

    real_avg_intervals = compute_avg_intervals(real_df_1)
    gen_avg_intervals = compute_avg_intervals(gen_df_2)

    # Convert to probability distributions
    real_dist = real_avg_intervals / real_avg_intervals.sum() if real_avg_intervals.sum() > 0 else np.zeros_like(real_avg_intervals)
    gen_dist = gen_avg_intervals / gen_avg_intervals.sum() if gen_avg_intervals.sum() > 0 else np.zeros_like(gen_avg_intervals)

    # Sort for port-agnostic comparison
    real_sorted = np.sort(real_dist)[::-1]
    gen_sorted = np.sort(gen_dist)[::-1]

    return jensenshannon_wrapper(real_sorted, gen_sorted, base=2)


def flow_srcport_stateful_flowduration_topnkey(real_df_1, gen_df_2, n=10):
    """
    Computes the 1 - hit rate between the top-N source ports (by flow duration)
    in the real and generated datasets. A flow is defined by srcport, and the duration
    is measured as the time span between the first and last packets in each srcport group.

    Args:
        real_df_1 (pd.DataFrame): Real dataset containing network packet data.
        gen_df_2 (pd.DataFrame): Generated dataset containing network packet data.
        n (int): Number of top source ports to consider.

    Returns:
        float: 1 - hit rate (i.e., distance) between real and generated top-N source ports by flow duration.
    """
    def compute_durations(df):
        valid = df.groupby('srcport').filter(lambda x: len(x) > 1)
        return valid.groupby('srcport')['time'].agg(lambda x: x.max() - x.min())

    # Compute top-N srcports by flow duration
    real_durations = compute_durations(real_df_1).nlargest(n)
    gen_durations = compute_durations(gen_df_2).nlargest(n)

    real_top_ports = set(real_durations.index)
    gen_top_ports = set(gen_durations.index)

    hits = len(real_top_ports.intersection(gen_top_ports))
    return 1 - hits / n


def flow_srcport_stateful_flowduration_topnvalue(real_df_1, gen_df_2, n=10):
    """
    Computes the average Absolute Relative Error (ARE) of the flow durations
    for the top N source ports (srcport) between the real and generated datasets.
    The port values are ignored; only the sorted top-N flow durations are compared.

    A flow's duration is defined as the time difference between the first and last
    packets associated with each srcport.

    Args:
        real_df_1 (pd.DataFrame): Real dataset containing network packet data.
        gen_df_2 (pd.DataFrame): Generated dataset containing network packet data.
        n (int): Number of top srcports to consider.

    Returns:
        float: Average ARE over the top-N flow durations, capped at 1.0.
    """
    import numpy as np

    def compute_durations(df):
        grouped = df.groupby('srcport').filter(lambda x: len(x) > 1).groupby('srcport')['time']
        return (grouped.max() - grouped.min()).values

    real_durations = compute_durations(real_df_1)
    gen_durations = compute_durations(gen_df_2)

    # Take top-N durations (sorted in descending order)
    real_top = np.sort(real_durations)[-n:][::-1]
    gen_top = np.sort(gen_durations)[-n:][::-1]

    # Pad arrays to equal length
    max_len = max(len(real_top), len(gen_top))
    real_top = np.pad(real_top, (0, max_len - len(real_top)))
    gen_top = np.pad(gen_top, (0, max_len - len(gen_top)))

    # Compute relative errors
    relative_errors = np.abs(real_top - gen_top) / np.maximum(real_top, 1)

    return min(relative_errors.mean(), 1.0)


def flow_srcport_stateful_flowduration_distribution(real_df_1, gen_df_2, n=10):
    """
    Computes the Jensen-Shannon Divergence (JSD) between the distributions
    of flow durations for each source port in the real and generated datasets.

    The port values are ignored; only the sorted distributions are compared.
    Only ports with more than one packet are considered.

    A flow's duration is defined as the time difference between the first and last
    packets associated with each srcport.

    Args:
        real_df_1 (pd.DataFrame): Real dataset containing network packet data.
        gen_df_2 (pd.DataFrame): Generated dataset containing network packet data.

    Returns:
        float: Jensen-Shannon Divergence (JSD) value in [0, 1].
    """
    import numpy as np

    def compute_durations(df):
        valid = df.groupby('srcport').filter(lambda x: len(x) > 1)
        grouped = valid.groupby('srcport')['time']
        return (grouped.max() - grouped.min()).values

    real_durations = compute_durations(real_df_1)
    gen_durations = compute_durations(gen_df_2)

    # Convert to probability distributions
    real_dist = real_durations / real_durations.sum() if real_durations.sum() > 0 else np.zeros_like(real_durations)
    gen_dist = gen_durations / gen_durations.sum() if gen_durations.sum() > 0 else np.zeros_like(gen_durations)

    # Sort distributions for port-agnostic comparison
    real_sorted = np.sort(real_dist)[::-1]
    gen_sorted = np.sort(gen_dist)[::-1]

    return jensenshannon_wrapper(real_sorted, gen_sorted, base=2)


def flow_srcport_stateful_byterate_topnkey(real_df_1, gen_df_2, n=10):
    """
    Computes the 1 - hit rate between the top-N source ports (by byte rate)
    in the real and generated datasets. Byte rate is defined as total bytes
    divided by flow duration for each source port. Only flows with more than
    one packet are considered.

    Args:
        real_df_1 (pd.DataFrame): Real dataset containing network packet data.
        gen_df_2 (pd.DataFrame): Generated dataset containing network packet data.
        n (int): Number of top source ports to consider.

    Returns:
        float: 1 - hit rate between real and generated top-N source ports by byte rate.
    """
    def compute_byterates(df):
        df = df.sort_values('time')
        valid = df.groupby('srcport').filter(lambda x: len(x) > 1)
        grouped = valid.groupby('srcport')
        bytes_total = grouped['pkt_len'].sum()
        durations = grouped['time'].agg(lambda x: x.max() - x.min())
        durations[durations == 0] = 1
        return (bytes_total / durations)

    real_rates = compute_byterates(real_df_1).nlargest(n)
    gen_rates = compute_byterates(gen_df_2).nlargest(n)

    real_top_ports = set(real_rates.index)
    gen_top_ports = set(gen_rates.index)

    hits = len(real_top_ports.intersection(gen_top_ports))
    return 1 - hits / n


def flow_srcport_stateful_byterate_topnvalue(real_df_1, gen_df_2, n=10):
    """
    Computes the average Absolute Relative Error (ARE) of byte rates (bytes per unit time)
    for the top N source ports in the real and generated datasets.
    Port values are ignored; only the top-N byte rates (sorted) are compared.

    Args:
        real_df_1 (pd.DataFrame): Real dataset containing network packet data.
        gen_df_2 (pd.DataFrame): Generated dataset containing network packet data.
        n (int): Number of top source ports to consider.

    Returns:
        float: Average ARE over the top-N byte rates, capped at 1.0.
    """
    import numpy as np

    def compute_byterates(df):
        grouped = df.groupby('srcport')
        valid = grouped.filter(lambda x: len(x) > 1).groupby('srcport')
        bytes_total = valid['pkt_len'].sum()
        durations = valid['time'].agg(lambda x: x.max() - x.min())
        durations[durations == 0] = 1  # avoid division by zero
        return bytes_total / durations

    # Compute byte rates and get top-N
    real_rates = compute_byterates(real_df_1).nlargest(n).values
    gen_rates = compute_byterates(gen_df_2).nlargest(n).values

    # Pad to equal length
    max_len = max(len(real_rates), len(gen_rates))
    real_rates = np.pad(real_rates, (0, max_len - len(real_rates)))
    gen_rates = np.pad(gen_rates, (0, max_len - len(gen_rates)))

    # Compute relative errors
    relative_errors = np.abs(real_rates - gen_rates) / np.maximum(real_rates, 1)
    relative_errors = np.minimum(relative_errors, 1.0)

    return min(relative_errors.mean(), 1.0)


def flow_srcport_stateful_byterate_distribution(real_df_1, gen_df_2, n=10):
    """
    Computes the Jensen-Shannon Divergence (JSD) between the distributions
    of byte rates (bytes per unit time) for each source port in the real and generated datasets.

    Port values are ignored; only the sorted distributions are compared.
    Only source ports with more than one packet are considered.

    Args:
        real_df_1 (pd.DataFrame): Real dataset containing network packet data.
        gen_df_2 (pd.DataFrame): Generated dataset containing network packet data.

    Returns:
        float: Jensen-Shannon Divergence (JSD) value in [0, 1].
    """
    import numpy as np

    def compute_byterates(df):
        grouped = df.groupby('srcport')
        valid = grouped.filter(lambda x: len(x) > 1).groupby('srcport')
        bytes_total = valid['pkt_len'].sum()
        durations = valid['time'].agg(lambda x: x.max() - x.min())
        durations[durations == 0] = 1  # avoid division by zero
        return (bytes_total / durations).values

    # Compute byte rate distributions
    real_rates = compute_byterates(real_df_1)
    gen_rates = compute_byterates(gen_df_2)

    # Convert to probability distributions
    real_dist = real_rates / real_rates.sum() if real_rates.sum() > 0 else np.zeros_like(real_rates)
    gen_dist = gen_rates / gen_rates.sum() if gen_rates.sum() > 0 else np.zeros_like(gen_rates)

    # Sort for port-agnostic comparison
    real_sorted = np.sort(real_dist)[::-1]
    gen_sorted = np.sort(gen_dist)[::-1]

    return jensenshannon_wrapper(real_sorted, gen_sorted, base=2)


def flow_dstport_stateless_packet_topnkey(real_df_1, gen_df_2, n=10):
    """
    Computes the 1 - hit rate between the top-N destination ports (by number of packets)
    in the real and generated datasets. This metric measures whether the same top-N ports
    appear in both datasets. Lower values indicate better match.

    Args:
        real_df_1 (pd.DataFrame): Real dataset containing network packet data.
        gen_df_2 (pd.DataFrame): Generated dataset containing network packet data.
        n (int): Number of top destination ports to consider.

    Returns:
        float: 1 - hit rate between real and generated top-N destination ports by packet count.
    """
    # Get top-N destination ports by packet count
    real_top_ports = set(real_df_1['dstport'].value_counts().nlargest(n).index)
    gen_top_ports = set(gen_df_2['dstport'].value_counts().nlargest(n).index)

    # Count hits
    hits = len(real_top_ports.intersection(gen_top_ports))

    return 1 - hits / n


def flow_dstport_stateless_packet_topnvalue(real_df_1, gen_df_2, n=10):
    """
    Computes the average Absolute Relative Error (ARE) of the number of packets
    received by the top N destination ports between the real and generated datasets.
    The actual port values are ignored; only the sorted top-N packet counts are compared.

    Args:
        real_df_1 (pd.DataFrame): Real dataset containing network packet data.
        gen_df_2 (pd.DataFrame): Generated dataset containing network packet data.
        n (int): Number of top destination ports to consider.

    Returns:
        float: Average ARE over the top-N destination ports, capped at 1.0.
    """
    import numpy as np

    real_counts = real_df_1['dstport'].value_counts().nlargest(n).values
    gen_counts = gen_df_2['dstport'].value_counts().nlargest(n).values

    max_len = max(len(real_counts), len(gen_counts))
    real_counts = np.pad(real_counts, (0, max_len - len(real_counts)))
    gen_counts = np.pad(gen_counts, (0, max_len - len(gen_counts)))

    relative_errors = np.abs(real_counts - gen_counts) / np.maximum(real_counts, 1)
    relative_errors = np.minimum(relative_errors, 1.0)

    return min(relative_errors.mean(), 1.0)


def flow_dstport_stateless_bytes_topnkey(real_df_1, gen_df_2, n=10):
    """
    Computes the 1 - hit rate between the top-N destination ports (by total bytes received)
    in the real and generated datasets. This measures how many of the real top-N ports
    are also present in the generated top-N ports.

    Args:
        real_df_1 (pd.DataFrame): Real dataset containing network packet data.
        gen_df_2 (pd.DataFrame): Generated dataset containing network packet data.
        n (int): Number of top destination ports to consider.

    Returns:
        float: 1 - hit rate between real and generated top-N destination ports by byte volume.
    """
    # Get top-N destination ports by byte volume
    real_top_ports = set(real_df_1.groupby('dstport')['pkt_len'].sum().nlargest(n).index)
    gen_top_ports = set(gen_df_2.groupby('dstport')['pkt_len'].sum().nlargest(n).index)

    # Count how many of the real top ports are in the generated top ports
    hits = len(real_top_ports.intersection(gen_top_ports))

    return 1 - hits / n


def flow_dstport_stateless_bytes_topnvalue(real_df_1, gen_df_2, n=10):
    """
    Computes the average Absolute Relative Error (ARE) of the number of bytes
    received by the top N destination ports between the real and generated datasets.
    The actual port values are ignored; only the sorted top-N byte volumes are compared.

    Args:
        real_df_1 (pd.DataFrame): Real dataset containing network packet data.
        gen_df_2 (pd.DataFrame): Generated dataset containing network packet data.
        n (int): Number of top destination ports to consider.

    Returns:
        float: Average ARE over the top-N destination ports, capped at 1.0.
    """
    import numpy as np

    # Get top-N byte volumes from both datasets
    real_bytes = real_df_1.groupby('dstport')['pkt_len'].sum().nlargest(n).values
    gen_bytes = gen_df_2.groupby('dstport')['pkt_len'].sum().nlargest(n).values

    # Pad arrays to equal length
    max_len = max(len(real_bytes), len(gen_bytes))
    real_bytes = np.pad(real_bytes, (0, max_len - len(real_bytes)))
    gen_bytes = np.pad(gen_bytes, (0, max_len - len(gen_bytes)))

    # Compute relative errors
    relative_errors = np.abs(real_bytes - gen_bytes) / np.maximum(real_bytes, 1)
    relative_errors = np.minimum(relative_errors, 1.0)

    return min(relative_errors.mean(), 1.0)


def flow_dstport_stateless_bytes_distribution(real_df_1, gen_df_2, n=10):
    """
    Computes the Jensen-Shannon Divergence (JSD) between the distribution of the number
    of bytes received by each destination port in the real and generated datasets.

    Args:
        real_df_1 (pd.DataFrame): Real dataset containing network packet data.
        gen_df_2 (pd.DataFrame): Generated dataset containing network packet data.

    Returns:
        float: Jensen-Shannon Divergence (JSD) value in [0, 1].
    """
    from scipy.spatial.distance import jensenshannon
    import numpy as np

    # Compute total bytes per destination port
    real_bytes = real_df_1.groupby('dstport')['pkt_len'].sum()
    gen_bytes = gen_df_2.groupby('dstport')['pkt_len'].sum()

    # Normalize to get probability distributions
    real_dist = real_bytes / real_bytes.sum() if real_bytes.sum() > 0 else np.zeros_like(real_bytes)
    gen_dist = gen_bytes / gen_bytes.sum() if gen_bytes.sum() > 0 else np.zeros_like(gen_bytes)

    # Align distributions by port
    all_ports = set(real_dist.index).union(set(gen_dist.index))
    real_aligned = real_dist.reindex(all_ports, fill_value=0).sort_index()
    gen_aligned = gen_dist.reindex(all_ports, fill_value=0).sort_index()

    return jensenshannon_wrapper(real_aligned.values, gen_aligned.values, base=2)


def flow_dstport_stateless_connection2dstip_topnkey(real_df_1, gen_df_2, n=10):
    """
    Computes the 1 - hit rate between the top-N destination ports (by number of distinct destination IPs)
    in the real and generated datasets. This measures how many of the real top-N ports are also
    present in the generated top-N ports. Lower values indicate greater distance.

    Args:
        real_df_1 (pd.DataFrame): Real dataset containing network packet data.
        gen_df_2 (pd.DataFrame): Generated dataset containing network packet data.
        n (int): Number of top destination ports to consider.

    Returns:
        float: 1 - hit rate between real and generated top-N destination ports.
    """
    # Identify top-N destination ports by number of distinct destination IPs
    real_top_ports = set(real_df_1.groupby('dstport')['dstip'].nunique().nlargest(n).index)
    gen_top_ports = set(gen_df_2.groupby('dstport')['dstip'].nunique().nlargest(n).index)

    # Count overlap between real and generated top ports
    hits = len(real_top_ports.intersection(gen_top_ports))

    return 1 - hits / n


def flow_dstport_stateless_connection2dstip_topnvalue(real_df_1, gen_df_2, n=10):
    """
    Computes the average Absolute Relative Error (ARE) of the number of distinct destination IPs
    each of the top N destination ports connected to in the real and generated datasets.
    The port numbers are ignored; only the sorted top-N connection counts are compared.

    Args:
        real_df_1 (pd.DataFrame): Real dataset containing network packet data.
        gen_df_2 (pd.DataFrame): Generated dataset containing network packet data.
        n (int): Number of top destination ports to consider.

    Returns:
        float: Average ARE over the top-N destination ports, capped at 1.0.
    """
    import numpy as np

    # Count distinct dstips per dstport
    real_conn = real_df_1.groupby('dstport')['dstip'].nunique()
    gen_conn = gen_df_2.groupby('dstport')['dstip'].nunique()

    # Extract top-N values (ignore port identities)
    real_top = real_conn.nlargest(n).values
    gen_top = gen_conn.nlargest(n).values

    # Pad with zeros to equal length
    max_len = max(len(real_top), len(gen_top))
    real_top = np.pad(real_top, (0, max_len - len(real_top)))
    gen_top = np.pad(gen_top, (0, max_len - len(gen_top)))

    # Compute relative errors
    relative_errors = np.abs(real_top - gen_top) / np.maximum(real_top, 1)
    relative_errors = np.minimum(relative_errors, 1.0)

    return min(relative_errors.mean(), 1.0)


def flow_dstport_stateless_connection2dstip_distribution(real_df_1, gen_df_2, n=10):
    """
    Computes the Jensen-Shannon Divergence (JSD) between the distribution of the number
    of distinct destination IPs connected to each destination port in the real and generated datasets.

    Args:
        real_df_1 (pd.DataFrame): Real dataset containing network packet data.
        gen_df_2 (pd.DataFrame): Generated dataset containing network packet data.

    Returns:
        float: Jensen-Shannon Divergence (JSD) value in [0, 1].
    """
    from scipy.spatial.distance import jensenshannon
    import numpy as np

    # Count number of distinct destination IPs per destination port
    real_conn = real_df_1.groupby('dstport')['dstip'].nunique()
    gen_conn = gen_df_2.groupby('dstport')['dstip'].nunique()

    # Normalize to probability distributions
    real_dist = real_conn / real_conn.sum() if real_conn.sum() > 0 else np.zeros_like(real_conn)
    gen_dist = gen_conn / gen_conn.sum() if gen_conn.sum() > 0 else np.zeros_like(gen_conn)

    # Align both distributions by destination port
    all_ports = set(real_dist.index).union(set(gen_conn.index))
    real_aligned = real_dist.reindex(all_ports, fill_value=0).sort_index()
    gen_aligned = gen_dist.reindex(all_ports, fill_value=0).sort_index()

    # Compute and return Jensen-Shannon Divergence
    return jensenshannon_wrapper(real_aligned.values, gen_aligned.values, base=2)


def flow_dstport_stateless_connection2srcip_topnkey(real_df_1, gen_df_2, n=10):
    """
    Computes the 1 - hit rate between the top-N destination ports (by number of distinct source IPs)
    in the real and generated datasets. This measures how many of the real top-N ports are also
    present in the generated top-N ports. Lower values indicate greater distance.

    Args:
        real_df_1 (pd.DataFrame): Real dataset containing network packet data.
        gen_df_2 (pd.DataFrame): Generated dataset containing network packet data.
        n (int): Number of top destination ports to consider.

    Returns:
        float: 1 - hit rate between real and generated top-N destination ports.
    """
    # Identify top-N destination ports by number of distinct source IPs
    real_top_ports = set(real_df_1.groupby('dstport')['srcip'].nunique().nlargest(n).index)
    gen_top_ports = set(gen_df_2.groupby('dstport')['srcip'].nunique().nlargest(n).index)

    # Count overlap between real and generated top ports
    hits = len(real_top_ports.intersection(gen_top_ports))

    return 1 - hits / n


def flow_dstport_stateless_connection2srcip_topnvalue(real_df_1, gen_df_2, n=10):
    """
    Computes the average Absolute Relative Error (ARE) of the number of distinct source IPs
    each of the top N destination ports connected to in the real and generated datasets.
    The port numbers are ignored; only the sorted top-N connection counts are compared.

    Args:
        real_df_1 (pd.DataFrame): Real dataset containing network packet data.
        gen_df_2 (pd.DataFrame): Generated dataset containing network packet data.
        n (int): Number of top destination ports to consider.

    Returns:
        float: Average ARE over the top-N destination ports, capped at 1.0.
    """
    import numpy as np

    # Count distinct srcip per dstport
    real_conn = real_df_1.groupby('dstport')['srcip'].nunique()
    gen_conn = gen_df_2.groupby('dstport')['srcip'].nunique()

    # Extract top-N values (ignore port identities)
    real_top = real_conn.nlargest(n).values
    gen_top = gen_conn.nlargest(n).values

    # Pad with zeros to equal length
    max_len = max(len(real_top), len(gen_top))
    real_top = np.pad(real_top, (0, max_len - len(real_top)))
    gen_top = np.pad(gen_top, (0, max_len - len(gen_top)))

    # Compute relative errors
    relative_errors = np.abs(real_top - gen_top) / np.maximum(real_top, 1)
    relative_errors = np.minimum(relative_errors, 1.0)

    return min(relative_errors.mean(), 1.0)


def flow_dstport_stateless_connection2srcip_distribution(real_df_1, gen_df_2, n=10):
    """
    Computes the Jensen-Shannon Divergence (JSD) between the distribution of the number
    of distinct source IPs connected to each destination port in the real and generated datasets.

    Args:
        real_df_1 (pd.DataFrame): Real dataset containing network packet data.
        gen_df_2 (pd.DataFrame): Generated dataset containing network packet data.

    Returns:
        float: Jensen-Shannon Divergence (JSD) value in [0, 1].
    """
    from scipy.spatial.distance import jensenshannon
    import numpy as np

    # Count number of distinct source IPs per destination port
    real_conn = real_df_1.groupby('dstport')['srcip'].nunique()
    gen_conn = gen_df_2.groupby('dstport')['srcip'].nunique()

    # Normalize to probability distributions
    real_dist = real_conn / real_conn.sum() if real_conn.sum() > 0 else np.zeros_like(real_conn)
    gen_dist = gen_conn / gen_conn.sum() if gen_conn.sum() > 0 else np.zeros_like(gen_conn)

    # Align both distributions by destination port
    all_ports = set(real_dist.index).union(set(gen_conn.index))
    real_aligned = real_dist.reindex(all_ports, fill_value=0).sort_index()
    gen_aligned = gen_dist.reindex(all_ports, fill_value=0).sort_index()

    # Compute and return Jensen-Shannon Divergence
    return jensenshannon_wrapper(real_aligned.values, gen_aligned.values, base=2)


def flow_dstport_stateless_connection2srcport_topnkey(real_df_1, gen_df_2, n=10):
    """
    Computes the 1 - hit rate between the top-N destination ports (by number of distinct source ports)
    in the real and generated datasets. This measures how many of the real top-N ports are also
    present in the generated top-N ports. Lower values indicate greater distance.

    Args:
        real_df_1 (pd.DataFrame): Real dataset containing network packet data.
        gen_df_2 (pd.DataFrame): Generated dataset containing network packet data.
        n (int): Number of top destination ports to consider.

    Returns:
        float: 1 - hit rate between real and generated top-N destination ports.
    """
    # Identify top-N destination ports by number of distinct source ports
    real_top_ports = set(real_df_1.groupby('dstport')['srcport'].nunique().nlargest(n).index)
    gen_top_ports = set(gen_df_2.groupby('dstport')['srcport'].nunique().nlargest(n).index)

    # Count overlap between real and generated top ports
    hits = len(real_top_ports.intersection(gen_top_ports))

    return 1 - hits / n


def flow_dstport_stateless_connection2srcport_topnvalue(real_df_1, gen_df_2, n=10):
    """
    Computes the average Absolute Relative Error (ARE) of the number of distinct source ports
    each of the top N destination ports connected to in the real and generated datasets.
    The port numbers themselves are ignored; only the sorted top-N connection counts are compared.

    Args:
        real_df_1 (pd.DataFrame): Real dataset containing network packet data.
        gen_df_2 (pd.DataFrame): Generated dataset containing network packet data.
        n (int): Number of top destination ports to consider.

    Returns:
        float: Average ARE over the top-N destination ports, capped at 1.0.
    """
    import numpy as np

    # Count distinct srcports per dstport
    real_conn = real_df_1.groupby('dstport')['srcport'].nunique()
    gen_conn = gen_df_2.groupby('dstport')['srcport'].nunique()

    # Extract top-N values, ignoring port identities
    real_top = real_conn.nlargest(n).values
    gen_top = gen_conn.nlargest(n).values

    # Pad with zeros to equal length
    max_len = max(len(real_top), len(gen_top))
    real_top = np.pad(real_top, (0, max_len - len(real_top)))
    gen_top = np.pad(gen_top, (0, max_len - len(gen_top)))

    # Compute absolute relative error
    relative_errors = np.abs(real_top - gen_top) / np.maximum(real_top, 1)
    relative_errors = np.minimum(relative_errors, 1.0)

    return min(relative_errors.mean(), 1.0)


def flow_dstport_stateless_connection2srcport_distribution(real_df_1, gen_df_2, n=10):
    """
    Computes the Jensen-Shannon Divergence (JSD) between the distribution of the number
    of distinct source ports connected to each destination port in the real and generated datasets.

    Args:
        real_df_1 (pd.DataFrame): Real dataset containing network packet data.
        gen_df_2 (pd.DataFrame): Generated dataset containing network packet data.

    Returns:
        float: Jensen-Shannon Divergence (JSD) value in [0, 1].
    """
    from scipy.spatial.distance import jensenshannon
    import numpy as np

    # Count number of distinct source ports per destination port
    real_conn = real_df_1.groupby('dstport')['srcport'].nunique()
    gen_conn = gen_df_2.groupby('dstport')['srcport'].nunique()

    # Normalize to probability distributions
    real_dist = real_conn / real_conn.sum() if real_conn.sum() > 0 else np.zeros_like(real_conn)
    gen_dist = gen_conn / gen_conn.sum() if gen_conn.sum() > 0 else np.zeros_like(gen_conn)

    # Align both distributions by destination port
    all_ports = set(real_dist.index).union(set(gen_conn.index))
    real_aligned = real_dist.reindex(all_ports, fill_value=0).sort_index()
    gen_aligned = gen_dist.reindex(all_ports, fill_value=0).sort_index()

    # Compute and return Jensen-Shannon Divergence
    return jensenshannon_wrapper(real_aligned.values, gen_aligned.values, base=2)


def flow_dstport_stateless_connection2srcipport_topnkey(real_df_1, gen_df_2, n=10):
    """
    Computes the 1 - hit rate between the top-N destination ports (by number of distinct (srcip, srcport) pairs)
    in the real and generated datasets. This measures how many of the real top-N ports are also
    present in the generated top-N ports. Lower values indicate greater distance.

    Args:
        real_df_1 (pd.DataFrame): Real dataset containing network packet data.
        gen_df_2 (pd.DataFrame): Generated dataset containing network packet data.
        n (int): Number of top destination ports to consider.

    Returns:
        float: 1 - hit rate between real and generated top-N destination ports.
    """
    # Count distinct (srcip, srcport) pairs per dstport
    real_conn = real_df_1.groupby('dstport').apply(lambda df: df[['srcip', 'srcport']].drop_duplicates().shape[0])
    gen_conn = gen_df_2.groupby('dstport').apply(lambda df: df[['srcip', 'srcport']].drop_duplicates().shape[0])

    # Get top-N dstports
    real_top_ports = set(real_conn.nlargest(n).index)
    gen_top_ports = set(gen_conn.nlargest(n).index)

    # Compute hit rate
    hits = len(real_top_ports.intersection(gen_top_ports))

    return 1 - hits / n


def flow_dstport_stateless_connection2srcipport_topnvalue(real_df_1, gen_df_2, n=10):
    """
    Computes the average Absolute Relative Error (ARE) of the number of distinct (srcip, srcport) pairs
    each of the top N destination ports connected to in the real and generated datasets.
    The port numbers themselves are ignored; only the sorted top-N connection counts are compared.

    Args:
        real_df_1 (pd.DataFrame): Real dataset containing network packet data.
        gen_df_2 (pd.DataFrame): Generated dataset containing network packet data.
        n (int): Number of top destination ports to consider.

    Returns:
        float: Average ARE over the top-N destination ports, capped at 1.0.
    """
    import numpy as np

    # Count distinct (srcip, srcport) pairs per dstport
    real_conn = real_df_1.groupby('dstport').apply(lambda df: df[['srcip', 'srcport']].drop_duplicates().shape[0])
    gen_conn = gen_df_2.groupby('dstport').apply(lambda df: df[['srcip', 'srcport']].drop_duplicates().shape[0])

    # Extract top-N values (ignore port identities)
    real_top = real_conn.nlargest(n).values
    gen_top = gen_conn.nlargest(n).values

    # Pad arrays to equal length
    max_len = max(len(real_top), len(gen_top))
    real_top = np.pad(real_top, (0, max_len - len(real_top)))
    gen_top = np.pad(gen_top, (0, max_len - len(gen_top)))

    # Compute absolute relative errors
    relative_errors = np.abs(real_top - gen_top) / np.maximum(real_top, 1)
    relative_errors = np.minimum(relative_errors, 1.0)

    return min(relative_errors.mean(), 1.0)


def flow_dstport_stateless_connection2srcipport_distribution(real_df_1, gen_df_2, n=10):
    """
    Computes the Jensen-Shannon Divergence (JSD) between the distribution of the number
    of distinct (srcip, srcport) pairs connected to each destination port in the real and generated datasets.

    Args:
        real_df_1 (pd.DataFrame): Real dataset containing network packet data.
        gen_df_2 (pd.DataFrame): Generated dataset containing network packet data.

    Returns:
        float: Jensen-Shannon Divergence (JSD) value in [0, 1].
    """
    from scipy.spatial.distance import jensenshannon
    import numpy as np

    # Count distinct (srcip, srcport) pairs per dstport
    real_conn = real_df_1.groupby('dstport').apply(lambda df: df[['srcip', 'srcport']].drop_duplicates().shape[0])
    gen_conn = gen_df_2.groupby('dstport').apply(lambda df: df[['srcip', 'srcport']].drop_duplicates().shape[0])

    # Normalize to probability distributions
    real_dist = real_conn / real_conn.sum() if real_conn.sum() > 0 else np.zeros_like(real_conn)
    gen_dist = gen_conn / gen_conn.sum() if gen_conn.sum() > 0 else np.zeros_like(gen_conn)

    # Align both distributions by dstport
    all_ports = set(real_dist.index).union(set(gen_dist.index))
    real_aligned = real_dist.reindex(all_ports, fill_value=0).sort_index()
    gen_aligned = gen_dist.reindex(all_ports, fill_value=0).sort_index()

    # Compute and return Jensen-Shannon Divergence
    return jensenshannon_wrapper(real_aligned.values, gen_aligned.values, base=2)


def flow_dstport_stateless_connection2flow_topnkey(real_df_1, gen_df_2, n=10):
    """
    Computes the 1 - hit rate between the top-N destination ports (by number of distinct flows)
    in the real and generated datasets. A flow is defined as a 5-tuple
    (srcip, dstip, srcport, dstport, proto). This metric checks overlap in top-N dstports.

    Args:
        real_df_1 (pd.DataFrame): Real dataset containing network packet data.
        gen_df_2 (pd.DataFrame): Generated dataset containing network packet data.
        n (int): Number of top destination ports to consider.

    Returns:
        float: 1 - hit rate between real and generated top-N destination ports by flow count.
    """
    # Define flows
    real_df_1 = real_df_1.copy()
    gen_df_2 = gen_df_2.copy()
    real_df_1['flow'] = list(zip(real_df_1['srcip'], real_df_1['dstip'], real_df_1['srcport'], real_df_1['dstport'], real_df_1['proto']))
    gen_df_2['flow'] = list(zip(gen_df_2['srcip'], gen_df_2['dstip'], gen_df_2['srcport'], gen_df_2['dstport'], gen_df_2['proto']))

    # Count unique flows per dstport
    real_counts = real_df_1.groupby('dstport')['flow'].nunique()
    gen_counts = gen_df_2.groupby('dstport')['flow'].nunique()

    # Get top-N dstports
    real_top = set(real_counts.nlargest(n).index)
    gen_top = set(gen_counts.nlargest(n).index)

    # Compute 1 - hit rate
    hits = len(real_top.intersection(gen_top))
    return 1 - hits / n


def flow_dstport_stateless_connection2flow_topnvalue(real_df_1, gen_df_2, n=10):
    """
    Computes the average Absolute Relative Error (ARE) of the number of distinct flows
    (defined as 5-tuples: srcip, dstip, srcport, dstport, proto) associated with each
    of the top N destination ports between the real and generated datasets.
    The port values are ignored; only the sorted top-N flow counts are compared.

    Args:
        real_df_1 (pd.DataFrame): Real dataset containing network packet data.
        gen_df_2 (pd.DataFrame): Generated dataset containing network packet data.
        n (int): Number of top destination ports to consider.

    Returns:
        float: Average ARE over the top-N destination ports, capped at 1.0.
    """
    import numpy as np

    def count_flows(df):
        df = df.copy()
        df['flow'] = list(zip(df['srcip'], df['dstip'], df['srcport'], df['dstport'], df['proto']))
        return df.groupby('dstport')['flow'].nunique()

    real_counts = count_flows(real_df_1)
    gen_counts = count_flows(gen_df_2)

    real_top = real_counts.sort_values(ascending=False).head(n).values
    gen_top = gen_counts.sort_values(ascending=False).head(n).values

    max_len = max(len(real_top), len(gen_top))
    real_top = np.pad(real_top, (0, max_len - len(real_top)))
    gen_top = np.pad(gen_top, (0, max_len - len(gen_top)))

    relative_errors = np.abs(real_top - gen_top) / np.maximum(real_top, 1)
    relative_errors = np.minimum(relative_errors, 1.0)

    return min(relative_errors.mean(), 1.0)


def flow_dstport_stateless_connection2flow_distribution(real_df_1, gen_df_2, n=10):
    """
    Computes the Jensen-Shannon Divergence (JSD) between the distribution of the number
    of distinct flows (defined as 5-tuples: srcip, dstip, srcport, dstport, proto)
    associated with each destination port in the real and generated datasets.

    Args:
        real_df_1 (pd.DataFrame): Real dataset containing network packet data.
        gen_df_2 (pd.DataFrame): Generated dataset containing network packet data.

    Returns:
        float: Jensen-Shannon Divergence (JSD) value in [0, 1].
    """
    from scipy.spatial.distance import jensenshannon
    import numpy as np

    def count_flows(df):
        df = df.copy()
        df['flow'] = list(zip(df['srcip'], df['dstip'], df['srcport'], df['dstport'], df['proto']))
        return df.groupby('dstport')['flow'].nunique()

    real_conn = count_flows(real_df_1)
    gen_conn = count_flows(gen_df_2)

    # Normalize to probability distributions
    real_dist = real_conn / real_conn.sum() if real_conn.sum() > 0 else np.zeros_like(real_conn)
    gen_dist = gen_conn / gen_conn.sum() if gen_conn.sum() > 0 else np.zeros_like(gen_conn)

    # Align by destination port
    all_ports = set(real_dist.index).union(set(gen_conn.index))
    real_aligned = real_dist.reindex(all_ports, fill_value=0).sort_index()
    gen_aligned = gen_dist.reindex(all_ports, fill_value=0).sort_index()

    return jensenshannon_wrapper(real_aligned.values, gen_aligned.values, base=2)


def flow_dstport_stateful_avgpacketinterval_topnkey(real_df_1, gen_df_2, n=10):
    """
    Computes the 1 - hit rate between the top-N destination ports (by average packet interval)
    in the real and generated datasets. A flow is defined by dstport. Only flows with more than
    one packet are considered.

    Args:
        real_df_1 (pd.DataFrame): Real dataset containing network packet data.
        gen_df_2 (pd.DataFrame): Generated dataset containing network packet data.
        n (int): Number of top destination ports to consider.

    Returns:
        float: 1 - hit rate between real and generated top-N destination ports by avg interval.
    """
    def compute_avg_intervals(df):
        df = df.sort_values('time')
        valid = df.groupby('dstport').filter(lambda x: len(x) > 1)
        return valid.groupby('dstport')['time'].agg(lambda x: np.diff(x).mean())

    real_avg = compute_avg_intervals(real_df_1).nlargest(n)
    gen_avg = compute_avg_intervals(gen_df_2).nlargest(n)

    real_top_ports = set(real_avg.index)
    gen_top_ports = set(gen_avg.index)

    hits = len(real_top_ports.intersection(gen_top_ports))
    return 1 - hits / n


def flow_dstport_stateful_avgpacketinterval_topnvalue(real_df_1, gen_df_2, n=10):
    """
    Computes the average Absolute Relative Error (ARE) of the average packet intervals
    for the top N destination ports (by number of packets) between the real and generated datasets.
    Destination port values are ignored; only the sorted average intervals are compared.

    Args:
        real_df_1 (pd.DataFrame): Real dataset containing network packet data.
        gen_df_2 (pd.DataFrame): Generated dataset containing network packet data.
        n (int): Number of top destination ports to consider.

    Returns:
        float: Average ARE over the top-N average intervals, capped at 1.0.
    """
    import numpy as np

    def compute_avg_intervals(df):
        valid = df.groupby('dstport').filter(lambda x: len(x) > 1)
        intervals = valid.sort_values('time').groupby('dstport')['time'].agg(lambda x: np.diff(x).mean())
        return intervals

    real_intervals = compute_avg_intervals(real_df_1).nlargest(n).values
    gen_intervals = compute_avg_intervals(gen_df_2).nlargest(n).values

    # Pad with zeros to equal length
    max_len = max(len(real_intervals), len(gen_intervals))
    real_intervals = np.pad(real_intervals, (0, max_len - len(real_intervals)))
    gen_intervals = np.pad(gen_intervals, (0, max_len - len(gen_intervals)))

    # Compute relative errors
    relative_errors = np.abs(real_intervals - gen_intervals) / np.maximum(real_intervals, 1)
    relative_errors = np.minimum(relative_errors, 1.0)

    return min(relative_errors.mean(), 1.0)


def flow_dstport_stateful_avgpacketinterval_distribution(real_df_1, gen_df_2, n=10):
    """
    Computes the Jensen-Shannon Divergence (JSD) between the distributions
    of average packet intervals for each destination port in the real and generated datasets.
    Port values are not agnostic; distributions are aligned by dstport.

    Only ports with more than one packet are considered.

    Args:
        real_df_1 (pd.DataFrame): Real dataset containing network packet data.
        gen_df_2 (pd.DataFrame): Generated dataset containing network packet data.

    Returns:
        float: Jensen-Shannon Divergence (JSD) value in [0, 1].
    """
    import numpy as np

    def compute_avg_intervals(df):
        df = df.sort_values('time')
        valid = df.groupby('dstport').filter(lambda x: len(x) > 1)
        return valid.groupby('dstport')['time'].agg(lambda x: np.diff(x).mean())

    real_intervals = compute_avg_intervals(real_df_1)
    gen_intervals = compute_avg_intervals(gen_df_2)

    # Align by dstport
    all_ports = set(real_intervals.index).union(set(gen_intervals.index))
    real_aligned = real_intervals.reindex(all_ports, fill_value=0).sort_index()
    gen_aligned = gen_intervals.reindex(all_ports, fill_value=0).sort_index()

    # Convert to distributions
    real_dist = real_aligned.values
    gen_dist = gen_aligned.values

    if real_dist.sum() > 0:
        real_dist = real_dist / real_dist.sum()
    else:
        real_dist = np.zeros_like(real_dist)

    if gen_dist.sum() > 0:
        gen_dist = gen_dist / gen_dist.sum()
    else:
        gen_dist = np.zeros_like(gen_dist)

    return jensenshannon_wrapper(real_dist, gen_dist, base=2)


def flow_dstport_stateful_flowduration_topnkey(real_df_1, gen_df_2, n=10):
    """
    Computes the 1 - hit rate between the top-N destination ports (by flow duration)
    in the real and generated datasets. A flow is defined by dstport, and the duration
    is measured as the time span between the first and last packets in each dstport group.

    Only flows with more than one packet are considered.

    Args:
        real_df_1 (pd.DataFrame): Real dataset containing network packet data.
        gen_df_2 (pd.DataFrame): Generated dataset containing network packet data.
        n (int): Number of top destination ports to consider.

    Returns:
        float: 1 - hit rate (i.e., distance) between real and generated top-N destination ports by flow duration.
    """
    def compute_durations(df):
        valid = df.groupby('dstport').filter(lambda x: len(x) > 1)
        return valid.groupby('dstport')['time'].agg(lambda x: x.max() - x.min())

    # Compute top-N dstports by flow duration
    real_durations = compute_durations(real_df_1).nlargest(n)
    gen_durations = compute_durations(gen_df_2).nlargest(n)

    real_top_ports = set(real_durations.index)
    gen_top_ports = set(gen_durations.index)

    hits = len(real_top_ports.intersection(gen_top_ports))
    return 1 - hits / n


def flow_dstport_stateful_flowduration_topnvalue(real_df_1, gen_df_2, n=10):
    """
    Computes the average Absolute Relative Error (ARE) of the flow durations
    for the top N destination ports (dstport) between the real and generated datasets.
    The port values are ignored; only the sorted top-N flow durations are compared.

    A flow's duration is defined as the time difference between the first and last
    packets associated with each dstport. Only flows with more than one packet are considered.

    Args:
        real_df_1 (pd.DataFrame): Real dataset containing network packet data.
        gen_df_2 (pd.DataFrame): Generated dataset containing network packet data.
        n (int): Number of top dstports to consider.

    Returns:
        float: Average ARE over the top-N flow durations, capped at 1.0.
    """
    import numpy as np

    def compute_durations(df):
        grouped = df.groupby('dstport').filter(lambda x: len(x) > 1).groupby('dstport')['time']
        return (grouped.max() - grouped.min()).values

    real_durations = compute_durations(real_df_1)
    gen_durations = compute_durations(gen_df_2)

    # Take top-N durations (sorted in descending order)
    real_top = np.sort(real_durations)[-n:][::-1]
    gen_top = np.sort(gen_durations)[-n:][::-1]

    # Pad arrays to equal length
    max_len = max(len(real_top), len(gen_top))
    real_top = np.pad(real_top, (0, max_len - len(real_top)))
    gen_top = np.pad(gen_top, (0, max_len - len(gen_top)))

    # Compute relative errors
    relative_errors = np.abs(real_top - gen_top) / np.maximum(real_top, 1)
    relative_errors = np.minimum(relative_errors, 1.0)

    return min(relative_errors.mean(), 1.0)


def flow_dstport_stateful_flowduration_distribution(real_df_1, gen_df_2, n=10):
    """
    Computes the Jensen-Shannon Divergence (JSD) between the distributions
    of flow durations for each destination port in the real and generated datasets.
    Port values are aligned (i.e., not port-agnostic).

    The flow duration is defined as the time difference between the first and last packets
    for each destination port. Only ports with more than one packet are considered.

    Returns:
        float: Jensen-Shannon Divergence (JSD) between the two flow duration distributions.
    """
    import numpy as np

    def compute_durations(df):
        df = df.sort_values('time')
        valid = df.groupby('dstport').filter(lambda x: len(x) > 1)
        grouped = valid.groupby('dstport')['time']
        return grouped.max() - grouped.min()

    real_durations = compute_durations(real_df_1)
    gen_durations = compute_durations(gen_df_2)

    # Align by dstport
    all_ports = set(real_durations.index).union(set(gen_durations.index))
    real_aligned = real_durations.reindex(all_ports, fill_value=0).sort_index()
    gen_aligned = gen_durations.reindex(all_ports, fill_value=0).sort_index()

    # Convert to probability distributions
    real_dist = real_aligned.values
    gen_dist = gen_aligned.values

    if real_dist.sum() > 0:
        real_dist = real_dist / real_dist.sum()
    else:
        real_dist = np.zeros_like(real_dist)

    if gen_dist.sum() > 0:
        gen_dist = gen_dist / gen_dist.sum()
    else:
        gen_dist = np.zeros_like(gen_dist)

    return jensenshannon_wrapper(real_dist, gen_dist, base=2)


def flow_dstport_stateful_byterate_topnkey(real_df_1, gen_df_2, n=10):
    """
    Computes the 1 - hit rate between the top-N destination ports (by byte rate)
    in the real and generated datasets. Byte rate is defined as total bytes
    divided by flow duration for each destination port. Only flows with more than
    one packet are considered.

    Args:
        real_df_1 (pd.DataFrame): Real dataset containing network packet data.
        gen_df_2 (pd.DataFrame): Generated dataset containing network packet data.
        n (int): Number of top destination ports to consider.

    Returns:
        float: 1 - hit rate between real and generated top-N destination ports by byte rate.
    """
    def compute_byterates(df):
        df = df.sort_values('time')
        valid = df.groupby('dstport').filter(lambda x: len(x) > 1)
        grouped = valid.groupby('dstport')
        bytes_total = grouped['pkt_len'].sum()
        durations = grouped['time'].agg(lambda x: x.max() - x.min())
        durations[durations == 0] = 1  # avoid division by zero
        return (bytes_total / durations)

    real_rates = compute_byterates(real_df_1).nlargest(n)
    gen_rates = compute_byterates(gen_df_2).nlargest(n)

    real_top_ports = set(real_rates.index)
    gen_top_ports = set(gen_rates.index)

    hits = len(real_top_ports.intersection(gen_top_ports))
    return 1 - hits / n


def flow_dstport_stateful_byterate_topnvalue(real_df_1, gen_df_2, n=10):
    """
    Computes the average Absolute Relative Error (ARE) of byte rates for the top-N destination ports
    (by byte rate) between the real and generated datasets. Byte rate is defined as total bytes
    divided by flow duration for each destination port. The destination port values themselves are ignored;
    only the sorted top-N byte rates are compared.

    Args:
        real_df_1 (pd.DataFrame): Real dataset containing network packet data.
        gen_df_2 (pd.DataFrame): Generated dataset containing network packet data.
        n (int): Number of top destination ports to consider.

    Returns:
        float: Average ARE over the top-N byte rates, capped at 1.0.
    """
    import numpy as np

    def compute_byterates(df):
        df = df.sort_values('time')
        valid = df.groupby('dstport').filter(lambda x: len(x) > 1)
        grouped = valid.groupby('dstport')
        bytes_total = grouped['pkt_len'].sum()
        durations = grouped['time'].agg(lambda x: x.max() - x.min())
        durations[durations == 0] = 1  # avoid division by zero
        return bytes_total / durations

    # Compute byte rates and get top-N
    real_rates = compute_byterates(real_df_1).nlargest(n).values
    gen_rates = compute_byterates(gen_df_2).nlargest(n).values

    # Pad to equal length
    max_len = max(len(real_rates), len(gen_rates))
    real_rates = np.pad(real_rates, (0, max_len - len(real_rates)))
    gen_rates = np.pad(gen_rates, (0, max_len - len(gen_rates)))

    # Compute relative errors
    relative_errors = np.abs(real_rates - gen_rates) / np.maximum(real_rates, 1)
    relative_errors = np.minimum(relative_errors, 1.0)

    return min(relative_errors.mean(), 1.0)


def flow_dstport_stateful_byterate_distribution(real_df_1, gen_df_2, n=10):
    """
    Computes the Jensen-Shannon Divergence (JSD) between the byte rate distributions
    for each destination port in the real and generated datasets.
    Byte rate is defined as total bytes divided by flow duration.
    Only flows with more than one packet are considered.
    Port values are aligned (not agnostic).

    Returns:
        float: Jensen-Shannon Divergence (JSD) between the two byte rate distributions.
    """
    import numpy as np

    def compute_byterates(df):
        df = df.sort_values('time')
        valid = df.groupby('dstport').filter(lambda x: len(x) > 1)
        grouped = valid.groupby('dstport')
        bytes_total = grouped['pkt_len'].sum()
        durations = grouped['time'].agg(lambda x: x.max() - x.min())
        durations[durations == 0] = 1  # avoid division by zero
        return bytes_total / durations

    # Compute byte rates
    real_rates = compute_byterates(real_df_1)
    gen_rates = compute_byterates(gen_df_2)

    # Align by dstport
    all_ports = set(real_rates.index).union(set(gen_rates.index))
    real_aligned = real_rates.reindex(all_ports, fill_value=0).sort_index()
    gen_aligned = gen_rates.reindex(all_ports, fill_value=0).sort_index()

    # Convert to distributions
    real_dist = real_aligned.values
    gen_dist = gen_aligned.values

    if real_dist.sum() > 0:
        real_dist = real_dist / real_dist.sum()
    else:
        real_dist = np.zeros_like(real_dist)

    if gen_dist.sum() > 0:
        gen_dist = gen_dist / gen_dist.sum()
    else:
        gen_dist = np.zeros_like(gen_dist)

    return jensenshannon_wrapper(real_dist, gen_dist, base=2)


def flow_fivetuple_stateless_packet_topnvalue(real_df_1, gen_df_2, n=10):
    """
    Computes the average Absolute Relative Error (ARE) of the number of packets per 5-tuple flow
    for the top-N flows in the real and generated datasets.

    Flows are defined by the 5-tuple (srcip, dstip, srcport, dstport, proto).
    Flow identities are ignored; only the top-N counts are compared (value-only).

    Args:
        real_df_1 (pd.DataFrame): Real dataset containing network packet data.
        gen_df_2 (pd.DataFrame): Generated dataset containing network packet data.
        n (int): Number of top flows to consider.

    Returns:
        float: Average ARE over the top-N flows, capped at 1.0.
    """
    import numpy as np

    # Count packets per 5-tuple flow
    real_counts = real_df_1.groupby(['srcip', 'dstip', 'srcport', 'dstport', 'proto']).size().nlargest(n).values
    gen_counts = gen_df_2.groupby(['srcip', 'dstip', 'srcport', 'dstport', 'proto']).size().nlargest(n).values

    # Pad to equal length
    max_len = max(len(real_counts), len(gen_counts))
    real_counts = np.pad(real_counts, (0, max_len - len(real_counts)))
    gen_counts = np.pad(gen_counts, (0, max_len - len(gen_counts)))

    # Compute relative errors
    relative_errors = np.abs(real_counts - gen_counts) / np.maximum(real_counts, 1)
    relative_errors = np.minimum(relative_errors, 1.0)

    return min(relative_errors.mean(), 1.0)


def flow_fivetuple_stateless_packet_distribution(real_df_1, gen_df_2, n=10):
    """
    Computes the Jensen-Shannon Divergence (JSD) between the packet count distributions
    of 5-tuple flows in the real and generated datasets. A 5-tuple flow is defined as
    (srcip, dstip, srcport, dstport, proto). The identities of the flows are ignored;
    only the sorted distribution values are compared (flow-agnostic).

    Args:
        real_df_1 (pd.DataFrame): Real dataset containing network packet data.
        gen_df_2 (pd.DataFrame): Generated dataset containing network packet data.

    Returns:
        float: Jensen-Shannon Divergence (JSD) value in [0, 1].
    """
    from scipy.spatial.distance import jensenshannon
    import numpy as np

    # Count packets per 5-tuple flow
    real_counts = real_df_1.groupby(['srcip', 'dstip', 'srcport', 'dstport', 'proto']).size().values
    gen_counts = gen_df_2.groupby(['srcip', 'dstip', 'srcport', 'dstport', 'proto']).size().values

    # Normalize to distributions
    real_dist = real_counts / real_counts.sum() if real_counts.sum() > 0 else np.zeros_like(real_counts)
    gen_dist = gen_counts / gen_counts.sum() if gen_counts.sum() > 0 else np.zeros_like(gen_counts)

    # Sort for flow-agnostic comparison
    real_sorted = np.sort(real_dist)[::-1]
    gen_sorted = np.sort(gen_dist)[::-1]

    # Pad to equal length
    max_len = max(len(real_sorted), len(gen_sorted))
    real_padded = np.pad(real_sorted, (0, max_len - len(real_sorted)))
    gen_padded = np.pad(gen_sorted, (0, max_len - len(gen_sorted)))

    return jensenshannon_wrapper(real_padded, gen_padded, base=2)


def flow_fivetuple_stateless_bytes_topnvalue(real_df_1, gen_df_2, n=10):
    """
    Computes the average Absolute Relative Error (ARE) of the number of bytes per 5-tuple flow
    for the top-N flows in the real and generated datasets.

    Flows are defined by the 5-tuple (srcip, dstip, srcport, dstport, proto).
    Flow identities are ignored; only the top-N byte counts are compared (value-only).

    Args:
        real_df_1 (pd.DataFrame): Real dataset containing network packet data.
        gen_df_2 (pd.DataFrame): Generated dataset containing network packet data.
        n (int): Number of top flows to consider.

    Returns:
        float: Average ARE over the top-N flows, capped at 1.0.
    """
    import numpy as np

    # Sum bytes per 5-tuple flow
    real_bytes = real_df_1.groupby(['srcip', 'dstip', 'srcport', 'dstport', 'proto'])['pkt_len'].sum().nlargest(n).values
    gen_bytes = gen_df_2.groupby(['srcip', 'dstip', 'srcport', 'dstport', 'proto'])['pkt_len'].sum().nlargest(n).values

    # Pad to equal length
    max_len = max(len(real_bytes), len(gen_bytes))
    real_bytes = np.pad(real_bytes, (0, max_len - len(real_bytes)))
    gen_bytes = np.pad(gen_bytes, (0, max_len - len(gen_bytes)))

    # Compute relative errors
    relative_errors = np.abs(real_bytes - gen_bytes) / np.maximum(real_bytes, 1)
    relative_errors = np.minimum(relative_errors, 1.0)

    return min(relative_errors.mean(), 1.0)


def flow_fivetuple_stateless_bytes_distribution(real_df_1, gen_df_2, n=10):
    """
    Computes the Jensen-Shannon Divergence (JSD) between the byte distributions
    of 5-tuple flows (srcip, dstip, srcport, dstport, proto) in the real and generated datasets.
    The identities of the flows are ignored; only the sorted distribution values are compared.

    Args:
        real_df_1 (pd.DataFrame): Real dataset containing network packet data.
        gen_df_2 (pd.DataFrame): Generated dataset containing network packet data.

    Returns:
        float: Jensen-Shannon Divergence (JSD) value in [0, 1].
    """
    from scipy.spatial.distance import jensenshannon
    import numpy as np

    # Sum bytes per 5-tuple flow
    real_bytes = real_df_1.groupby(['srcip', 'dstip', 'srcport', 'dstport', 'proto'])['pkt_len'].sum().values
    gen_bytes = gen_df_2.groupby(['srcip', 'dstip', 'srcport', 'dstport', 'proto'])['pkt_len'].sum().values

    # Normalize to probability distributions
    real_dist = real_bytes / real_bytes.sum() if real_bytes.sum() > 0 else np.zeros_like(real_bytes)
    gen_dist = gen_bytes / gen_bytes.sum() if gen_bytes.sum() > 0 else np.zeros_like(gen_bytes)

    # Sort distributions for flow-agnostic comparison
    real_sorted = np.sort(real_dist)[::-1]
    gen_sorted = np.sort(gen_dist)[::-1]

    # Pad to equal length
    max_len = max(len(real_sorted), len(gen_sorted))
    real_padded = np.pad(real_sorted, (0, max_len - len(real_sorted)))
    gen_padded = np.pad(gen_sorted, (0, max_len - len(gen_sorted)))

    return jensenshannon_wrapper(real_padded, gen_padded, base=2)


def flow_fivetuple_stateful_avgpacketinterval_topnvalue(real_df_1, gen_df_2, n=10):
    """
    Computes the average Absolute Relative Error (ARE) of the average packet intervals
    for the top-N 5-tuple flows (srcip, dstip, srcport, dstport, proto) between the real and generated datasets.

    Only flows with more than one packet are considered.
    Flow identities are ignored; only the top-N average intervals are compared (value-only).

    Args:
        real_df_1 (pd.DataFrame): Real dataset containing network packet data.
        gen_df_2 (pd.DataFrame): Generated dataset containing network packet data.
        n (int): Number of top flows to consider.

    Returns:
        float: Average ARE over the top-N flows, capped at 1.0.
    """
    import numpy as np

    def compute_avg_intervals(df):
        # Group by 5-tuple
        grouped = df.groupby(['srcip', 'dstip', 'srcport', 'dstport', 'proto'])
        # Filter flows with more than one packet and compute mean inter-packet time
        valid = grouped.filter(lambda x: len(x) > 1)
        intervals = valid.sort_values('time').groupby(['srcip', 'dstip', 'srcport', 'dstport', 'proto'])['time'].agg(
            lambda x: np.diff(x).mean())
        return intervals

    # Compute average packet intervals
    real_intervals = compute_avg_intervals(real_df_1).nlargest(n).values
    gen_intervals = compute_avg_intervals(gen_df_2).nlargest(n).values

    # Pad to equal length
    max_len = max(len(real_intervals), len(gen_intervals))
    real_intervals = np.pad(real_intervals, (0, max_len - len(real_intervals)))
    gen_intervals = np.pad(gen_intervals, (0, max_len - len(gen_intervals)))

    # Compute relative errors
    relative_errors = np.abs(real_intervals - gen_intervals) / np.maximum(real_intervals, 1)
    relative_errors = np.minimum(relative_errors, 1.0)

    return min(relative_errors.mean(), 1.0)


def flow_fivetuple_stateful_avgpacketinterval_distribution(real_df_1, gen_df_2, n=10):
    """
    Computes the Jensen-Shannon Divergence (JSD) between the average packet interval distributions
    of 5-tuple flows (srcip, dstip, srcport, dstport, proto) in the real and generated datasets.

    Only flows with more than one packet are considered.
    Flow identities are ignored; only the sorted distributions are compared.

    Returns:
        float: Jensen-Shannon Divergence (JSD) value in [0, 1].
    """
    import numpy as np

    def compute_avg_intervals(df):
        grouped = df.groupby(['srcip', 'dstip', 'srcport', 'dstport', 'proto'])
        valid = grouped.filter(lambda x: len(x) > 1)
        return valid.sort_values('time') \
                    .groupby(['srcip', 'dstip', 'srcport', 'dstport', 'proto'])['time'] \
                    .agg(lambda x: np.diff(x).mean()).values

    real_intervals = compute_avg_intervals(real_df_1)
    gen_intervals = compute_avg_intervals(gen_df_2)

    # Convert to distributions (sort and normalize)
    real_sorted = np.sort(real_intervals)[::-1]
    gen_sorted = np.sort(gen_intervals)[::-1]

    if real_sorted.sum() > 0:
        real_dist = real_sorted / real_sorted.sum()
    else:
        real_dist = np.zeros_like(real_sorted)

    if gen_sorted.sum() > 0:
        gen_dist = gen_sorted / gen_sorted.sum()
    else:
        gen_dist = np.zeros_like(gen_sorted)

    return jensenshannon_wrapper(real_dist, gen_dist, base=2)


def flow_fivetuple_stateful_flowduration_topnvalue(real_df_1, gen_df_2, n=10):
    """
    Computes the average Absolute Relative Error (ARE) of the flow durations
    for the top N 5-tuple flows between the real and generated datasets.

    A flow is defined as the 5-tuple: (srcip, dstip, srcport, dstport, proto).
    Flow identities are ignored; only the sorted top-N durations are compared.

    Returns:
        float: Average ARE over the top-N flows, capped at 1.0.
    """
    import numpy as np

    def compute_durations(df):
        grouped = df.groupby(['srcip', 'dstip', 'srcport', 'dstport', 'proto'])['time']
        return (grouped.max() - grouped.min()).values

    real_durations = compute_durations(real_df_1)
    gen_durations = compute_durations(gen_df_2)

    # Take top-N durations (descending)
    real_top = np.sort(real_durations)[-n:][::-1]
    gen_top = np.sort(gen_durations)[-n:][::-1]

    # Pad arrays to equal length
    max_len = max(len(real_top), len(gen_top))
    real_top = np.pad(real_top, (0, max_len - len(real_top)))
    gen_top = np.pad(gen_top, (0, max_len - len(gen_top)))

    # Compute relative errors
    relative_errors = np.abs(real_top - gen_top) / np.maximum(real_top, 1)
    relative_errors = np.minimum(relative_errors, 1.0)

    return min(relative_errors.mean(), 1.0)


def flow_fivetuple_stateful_flowduration_distribution(real_df_1, gen_df_2, n=10):
    """
    Computes the Jensen-Shannon Divergence (JSD) between the normalized flow duration
    distributions for each 5-tuple flow in the real and generated datasets.

    A flow is defined as the 5-tuple: (srcip, dstip, srcport, dstport, proto).
    Flow identities are ignored; only the sorted duration distributions are compared.

    Returns:
        float: Jensen-Shannon Divergence (JSD) value in [0, 1].
    """
    from scipy.spatial.distance import jensenshannon
    import numpy as np

    def compute_durations(df):
        grouped = df.groupby(['srcip', 'dstip', 'srcport', 'dstport', 'proto'])['time']
        return (grouped.max() - grouped.min()).values

    real_durations = compute_durations(real_df_1)
    gen_durations = compute_durations(gen_df_2)

    # Normalize to form distributions
    real_dist = real_durations / real_durations.sum() if real_durations.sum() > 0 else np.zeros_like(real_durations)
    gen_dist = gen_durations / gen_durations.sum() if gen_durations.sum() > 0 else np.zeros_like(gen_durations)

    # Sort for flow-agnostic comparison
    real_sorted = np.sort(real_dist)[::-1]
    gen_sorted = np.sort(gen_dist)[::-1]

    # Pad to equal length
    max_len = max(len(real_sorted), len(gen_sorted))
    real_padded = np.pad(real_sorted, (0, max_len - len(real_sorted)))
    gen_padded = np.pad(gen_sorted, (0, max_len - len(gen_sorted)))

    return jensenshannon_wrapper(real_padded, gen_padded, base=2)


def flow_fivetuple_stateful_byterate_topnvalue(real_df_1, gen_df_2, n=10):
    """
    Computes the average Absolute Relative Error (ARE) of byte rates (bytes per unit time)
    for the top N 5-tuple flows in the real and generated datasets.

    A 5-tuple flow is defined as (srcip, dstip, srcport, dstport, proto).
    Byte rate is defined as total bytes divided by flow duration.
    Flow identities are ignored; only the sorted top-N byte rates are compared.

    Args:
        real_df_1 (pd.DataFrame): Real dataset containing network packet data.
        gen_df_2 (pd.DataFrame): Generated dataset containing network packet data.
        n (int): Number of top flows to consider.

    Returns:
        float: Average ARE over the top-N byte rates, capped at 1.0.
    """
    import numpy as np

    def compute_byterates(df):
        grouped = df.groupby(['srcip', 'dstip', 'srcport', 'dstport', 'proto'])
        pkt_sum = grouped['pkt_len'].sum()
        time_range = grouped['time'].agg(lambda x: x.max() - x.min())
        time_range[time_range == 0] = 1  # prevent division by zero
        return (pkt_sum / time_range)

    real_rates = compute_byterates(real_df_1).nlargest(n).values
    gen_rates = compute_byterates(gen_df_2).nlargest(n).values

    # Pad to equal length
    max_len = max(len(real_rates), len(gen_rates))
    real_rates = np.pad(real_rates, (0, max_len - len(real_rates)))
    gen_rates = np.pad(gen_rates, (0, max_len - len(gen_rates)))

    relative_errors = np.abs(real_rates - gen_rates) / np.maximum(real_rates, 1)
    relative_errors = np.minimum(relative_errors, 1.0)

    return min(relative_errors.mean(), 1.0)


def flow_fivetuple_stateful_byterate_distribution(real_df_1, gen_df_2, n=10):
    """
    Computes the Jensen-Shannon Divergence (JSD) between the byte rate distributions
    of 5-tuple flows (srcip, dstip, srcport, dstport, proto) in the real and generated datasets.
    
    Byte rate is defined as total bytes divided by flow duration.
    Only flows with more than one packet are considered.
    Flow identities are ignored; only the sorted distributions are compared.

    Returns:
        float: Jensen-Shannon Divergence (JSD) value in [0, 1].
    """
    import numpy as np

    def compute_byterates(df):
        grouped = df.groupby(['srcip', 'dstip', 'srcport', 'dstport', 'proto'])
        valid = grouped.filter(lambda x: len(x) > 1).groupby(['srcip', 'dstip', 'srcport', 'dstport', 'proto'])
        pkt_sum = valid['pkt_len'].sum()
        durations = valid['time'].agg(lambda x: x.max() - x.min())
        durations[durations == 0] = 1  # avoid division by zero
        return (pkt_sum / durations).values

    real_rates = compute_byterates(real_df_1)
    gen_rates = compute_byterates(gen_df_2)

    real_sorted = np.sort(real_rates)[::-1]
    gen_sorted = np.sort(gen_rates)[::-1]

    if real_sorted.sum() > 0:
        real_dist = real_sorted / real_sorted.sum()
    else:
        real_dist = np.zeros_like(real_sorted)

    if gen_sorted.sum() > 0:
        gen_dist = gen_sorted / gen_sorted.sum()
    else:
        gen_dist = np.zeros_like(gen_sorted)

    return jensenshannon_wrapper(real_dist, gen_dist, base=2)


def eval_metrics(real_df, gen_df, n=10, n_threads=None):
    """
    Evaluates an exhaustive list of metric functions defined in metric.py and returns a dictionary
    with the computed scores grouped by metric type.
    ...
    Args:
        real_df (pd.DataFrame): The real dataset containing network packet data.
        gen_df (pd.DataFrame): The generated dataset containing network packet data.
        n (int, optional): The "top-N" parameter for metrics that require it (default: 10).
        n_threads (int, optional): Number of threads to use for parallel execution. 
                                   If None, defaults to min(32, os.cpu_count()).
    """
    import os
    import numpy as np
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _call_metric(func):
        return func.__name__, func(real_df, gen_df, n)

    def _run_group(funcs, desc):
        results = {}
        max_workers = n_threads if n_threads is not None else max(1, min(32, (os.cpu_count() or 4)))
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(_call_metric, f) for f in funcs]
            for fut in tqdm(as_completed(futures), total=len(futures), desc=desc, unit="metric"):
                name, value = fut.result()
                results[name] = value
        return results

    # Exhaustive lists of metric functions (update these lists as your module grows)
    packet_funcs = [
        packet_stateless__count,
        packet_stateless_srcip_countdistinct,
        packet_stateless_srcip_distribution,
        packet_stateless_dstip_countdistinct,
        packet_stateless_dstip_distribution,
        packet_stateless_srcport_countdistinct,
        packet_stateless_srcport_distribution,
        packet_stateless_dstport_countdistinct,
        packet_stateless_dstport_distribution,
        packet_stateless_proto_countdistinct,
        packet_stateless_proto_distribution,
        packet_stateless_time_distribution,
        packet_stateless_pktlen_sum,
        packet_stateless_pktlen_avg,
        packet_stateless_pktlen_distribution,
        # packet_stateless_flag_countdistinct,
        # packet_stateless_flag_distribution,
        # packet_stateless_ttl_avg,
        # packet_stateless_ttl_distribution
    ]
    
    flow_stateless_funcs = [
        flow_srcip_stateless_packet_topnvalue,
        flow_srcip_stateless_bytes_topnvalue,
        flow_srcip_stateless_bytes_distribution,
        flow_srcip_stateless_connection2srcport_topnvalue,
        flow_srcip_stateless_connection2srcport_distribution,
        flow_srcip_stateless_connection2dstip_topnvalue,
        flow_srcip_stateless_connection2dstip_distribution,
        flow_srcip_stateless_connection2dstport_topnvalue,
        flow_srcip_stateless_connection2dstport_distribution,
        flow_srcip_stateless_connection2dstipport_topnvalue,
        flow_srcip_stateless_connection2dstipport_distribution,
        flow_srcip_stateless_connection2flow_topnvalue,
        flow_srcip_stateless_connection2flow_distribution,
        flow_dstip_stateless_packet_topnvalue,
        flow_dstip_stateless_bytes_topnvalue,
        flow_dstip_stateless_bytes_distribution,
        flow_dstip_stateless_connection2dstport_topnvalue,
        flow_dstip_stateless_connection2dstport_distribution,
        flow_dstip_stateless_connection2srcip_topnvalue,
        flow_dstip_stateless_connection2srcip_distribution,
        flow_dstip_stateless_connection2srcport_topnvalue,
        flow_dstip_stateless_connection2srcport_distribution,
        flow_dstip_stateless_connection2srcipport_topnvalue,
        flow_dstip_stateless_connection2srcipport_distribution,
        flow_dstip_stateless_connection2flow_topnvalue,
        flow_dstip_stateless_connection2flow_distribution,
        flow_ippair_stateless_packet_topnvalue,
        flow_ippair_stateless_packet_distribution,
        flow_ippair_stateless_bytes_topnvalue,
        flow_ippair_stateless_bytes_distribution,
        flow_ippair_stateless_connection2srcport_topnvalue,
        flow_ippair_stateless_connection2srcport_distribution,
        flow_ippair_stateless_connection2dstport_topnvalue,
        flow_ippair_stateless_connection2dstport_distribution,
        flow_ippair_stateless_connection2flow_topnvalue,
        flow_ippair_stateless_connection2flow_distribution,
        flow_srcport_stateless_packet_topnkey,
        flow_srcport_stateless_packet_topnvalue,
        flow_srcport_stateless_bytes_topnkey,
        flow_srcport_stateless_bytes_topnvalue,
        flow_srcport_stateless_bytes_distribution,
        flow_srcport_stateless_connection2srcip_topnkey,
        flow_srcport_stateless_connection2srcip_topnvalue,
        flow_srcport_stateless_connection2srcip_distribution,
        flow_srcport_stateless_connection2dstip_topnkey,
        flow_srcport_stateless_connection2dstip_topnvalue,
        flow_srcport_stateless_connection2dstip_distribution,
        flow_srcport_stateless_connection2dstport_topnkey,
        flow_srcport_stateless_connection2dstport_topnvalue,
        flow_srcport_stateless_connection2dstport_distribution,
        flow_srcport_stateless_connection2dstipport_topnkey,
        flow_srcport_stateless_connection2dstipport_topnvalue,
        flow_srcport_stateless_connection2dstipport_distribution,
        flow_srcport_stateless_connection2flow_topnkey,
        flow_srcport_stateless_connection2flow_topnvalue,
        flow_srcport_stateless_connection2flow_distribution,
        flow_dstport_stateless_packet_topnkey,
        flow_dstport_stateless_packet_topnvalue,
        flow_dstport_stateless_bytes_topnkey,
        flow_dstport_stateless_bytes_topnvalue,
        flow_dstport_stateless_bytes_distribution,
        flow_dstport_stateless_connection2dstip_topnkey,
        flow_dstport_stateless_connection2dstip_topnvalue,
        flow_dstport_stateless_connection2dstip_distribution,
        flow_dstport_stateless_connection2srcip_topnkey,
        flow_dstport_stateless_connection2srcip_topnvalue,
        flow_dstport_stateless_connection2srcip_distribution,
        flow_dstport_stateless_connection2srcport_topnkey,
        flow_dstport_stateless_connection2srcport_topnvalue,
        flow_dstport_stateless_connection2srcport_distribution,
        flow_dstport_stateless_connection2srcipport_topnkey,
        flow_dstport_stateless_connection2srcipport_topnvalue,
        flow_dstport_stateless_connection2srcipport_distribution,
        flow_dstport_stateless_connection2flow_topnkey,
        flow_dstport_stateless_connection2flow_topnvalue,
        flow_dstport_stateless_connection2flow_distribution,
        flow_fivetuple_stateless_packet_topnvalue,
        flow_fivetuple_stateless_packet_distribution,
        flow_fivetuple_stateless_bytes_topnvalue,
        flow_fivetuple_stateless_bytes_distribution
    ]
    
    flow_stateful_funcs = [
        flow_srcip_stateful_avgpacketinterval_topnvalue,
        flow_srcip_stateful_avgpacketinterval_distribution,
        flow_srcip_stateful_flowduration_topnvalue,
        flow_srcip_stateful_flowduration_distribution,
        flow_srcip_stateful_byterate_topnvalue,
        flow_srcip_stateful_byterate_distribution,
        flow_dstip_stateful_avgpacketinterval_topnvalue,
        flow_dstip_stateful_avgpacketinterval_distribution,
        flow_dstip_stateful_flowduration_topnvalue,
        flow_dstip_stateful_flowduration_distribution,
        flow_dstip_stateful_byterate_topnvalue,
        flow_dstip_stateful_byterate_distribution,
        flow_ippair_stateful_avgpacketinterval_topnvalue,
        flow_ippair_stateful_avgpacketinterval_distribution,
        flow_ippair_stateful_flowduration_topnvalue,
        flow_ippair_stateful_flowduration_distribution,
        flow_ippair_stateful_byterate_topnvalue,
        flow_ippair_stateful_byterate_distribution,
        flow_srcport_stateful_avgpacketinterval_topnkey,
        flow_srcport_stateful_avgpacketinterval_topnvalue,
        flow_srcport_stateful_avgpacketinterval_distribution,
        flow_srcport_stateful_flowduration_topnkey,
        flow_srcport_stateful_flowduration_topnvalue,
        flow_srcport_stateful_flowduration_distribution,
        flow_srcport_stateful_byterate_topnkey,
        flow_srcport_stateful_byterate_topnvalue,
        flow_srcport_stateful_byterate_distribution,
        flow_dstport_stateful_avgpacketinterval_topnkey,
        flow_dstport_stateful_avgpacketinterval_topnvalue,
        flow_dstport_stateful_avgpacketinterval_distribution,
        flow_dstport_stateful_flowduration_topnkey,
        flow_dstport_stateful_flowduration_topnvalue,
        flow_dstport_stateful_flowduration_distribution,
        flow_dstport_stateful_byterate_topnkey,
        flow_dstport_stateful_byterate_topnvalue,
        flow_dstport_stateful_byterate_distribution,
        flow_fivetuple_stateful_avgpacketinterval_topnvalue,
        flow_fivetuple_stateful_avgpacketinterval_distribution,
        flow_fivetuple_stateful_flowduration_topnvalue,
        flow_fivetuple_stateful_flowduration_distribution,
        flow_fivetuple_stateful_byterate_topnvalue,
        flow_fivetuple_stateful_byterate_distribution
    ]
    
    results = {
        "packet": {"avg": None, "score": {}},
        "flow_stateless": {"avg": None, "score": {}},
        "flow_stateful": {"avg": None, "score": {}}
    }

    # Evaluate groups in parallel per group (each group's metrics are parallelized internally)
    packet_scores = _run_group(packet_funcs, desc="Evaluating Packet Metrics")
    results["packet"]["score"].update(packet_scores)
    packet_values = list(packet_scores.values())
    results["packet"]["avg"] = float(np.mean(packet_values)) if packet_values else None

    stateless_scores = _run_group(flow_stateless_funcs, desc="Evaluating Flow Stateless Metrics")
    results["flow_stateless"]["score"].update(stateless_scores)
    stateless_values = list(stateless_scores.values())
    results["flow_stateless"]["avg"] = float(np.mean(stateless_values)) if stateless_values else None

    stateful_scores = _run_group(flow_stateful_funcs, desc="Evaluating Flow Stateful Metrics")
    results["flow_stateful"]["score"].update(stateful_scores)
    stateful_values = list(stateful_scores.values())
    results["flow_stateful"]["avg"] = float(np.mean(stateful_values)) if stateful_values else None

    return results


def batch_eval(
    real_csv_path: str,
    gen_csv_list_path: str,
    n: int = 10,
    n_threads: int = None
):
    """
    Batch-evaluate generated datasets listed in a file against a single real dataset.

    The file at ``gen_csv_list_path`` should contain one entry per line. Each line can be:
      - a single CSV path, e.g. ``/path/to/gen.csv``
      - multiple CSV paths separated by commas, e.g. ``/p/a.csv,/p/b.csv``

    For a single path, this calls ``read_network_packets(path)`` and then ``eval_metrics``.
    For multiple paths, this calls ``read_network_packets((path1, path2, ...))`` (tuple) and then ``eval_metrics``.

    Args:
        real_csv_path: Path to the real dataset CSV.
        gen_csv_list_path: Path to a text file that lists one or more generated CSV paths per line.
        n: Top-N parameter for metrics that require it.
        n_threads: Number of threads to use inside ``eval_metrics``.

    Returns:
        List[dict]: Each dict contains:
            - "paths": list of paths evaluated for that line (one or many)
            - "packet_avg": float or None
            - "flow_stateless_avg": float or None
            - "flow_stateful_avg": float or None
            - "metrics": full result dict returned by ``eval_metrics``
    """
    results = []

    # Load the real dataset once
    real_df = read_network_packets(real_csv_path)

    if not os.path.isfile(gen_csv_list_path):
        raise FileNotFoundError(f"gen_csv_list_path not found: {gen_csv_list_path}")

    with open(gen_csv_list_path, "r") as f:
        for lineno, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                # Skip empty/comment lines
                continue

            # Split by comma; treat single vs multiple accordingly
            parts = [p.strip() for p in line.split(',') if p.strip()]
            gen_arg = parts[0] if len(parts) == 1 else tuple(parts)

            print(f"[batch_eval] Line {lineno}: Evaluating paths: {gen_arg}")
            # Basic existence check for paths
            if isinstance(gen_arg, tuple):
                missing = [p for p in gen_arg if not os.path.isfile(p)]
                if missing:
                    print(f"[batch_eval] Line {lineno}: skipping due to missing files: {missing}")
                    continue
            else:
                if not os.path.isfile(gen_arg):
                    print(f"[batch_eval] Line {lineno}: skipping due to missing file: {gen_arg}")
                    continue

            # Load generated data
            try:
                gen_df = read_network_packets(gen_arg)
            except Exception as e:
                if isinstance(gen_arg, tuple):
                    print(f"[batch_eval] Line {lineno}: read_network_packets failed for tuple: {e}. Skipping.")

            # Evaluate metrics
            rst = eval_metrics(real_df, gen_df, n=n, n_threads=n_threads)

            results.append({
                "paths": list(gen_arg) if isinstance(gen_arg, tuple) else [gen_arg],
                "packet_avg": rst["packet"]["avg"],
                "flow_stateless_avg": rst["flow_stateless"]["avg"],
                "flow_stateful_avg": rst["flow_stateful"]["avg"],
                "metrics": rst,
            })

    return results
    


if __name__ == "__main__":
    csv_path1 = '/home/steven/Projects/DeepStore/data/small-scale/caida/caida-first-1000000.csv'
    # csv_path2 = '/home/steven/Projects/DeepStore/results-long/small-scale/runs/realtabformer-tabular_caida2_20250513033548072/rtf_checkpoints/gen_data_60.csv'
    csv_path2 = '/home/steven/Projects/DeepStore/results-long/small-scale/csv/tvae_caida_20250919142741787_epoch_1.csv'

    real_df = pd.read_csv(csv_path1)
    gen_df = pd.read_csv(csv_path2)

    rst = eval_metrics(real_df, gen_df, n=10, n_threads=16)

    print("==================================")
    print(rst['packet'])
    print("==================================")
    print(rst['flow_stateless'])
    print("==================================")
    print(rst['flow_stateful'])
    print("==================================")

    # print(flow_fivetuple_stateful_flowduration_distribution(real_df, gen_df))


    # csv_path2_prefix = "/home/steven/Projects/DeepStore/results-long/small-scale/runs/realtabformer-tabular_caida2_20241230170520572/rtf_checkpoints/gen_data_"
    # print(csv_path2_prefix)

    # for e in range(5,61,5):
    #     csv_path2 = f"{csv_path2_prefix}{e}.csv"
    #     gen_df = read_network_packets(csv_path2)
    #     rst = eval_metrics(real_df, gen_df, n=10)
    #     avg_packet = rst["packet"]["avg"]
    #     avg_stateless = rst["flow_stateless"]["avg"]
    #     avg_stateful = rst["flow_stateful"]["avg"]
    #     print(f"EPOCH {e}")
    #     print(f"PacketAvg: {avg_packet}")
    #     print(f"StatelessAvg: {avg_stateless}")
    #     print(f"StatefulAvg: {avg_stateful}")
    #     print("--------------------------------------------------")



    # # Path to real CSV
    # csv_path_real = '/ocean/projects/cis230086p/sdong3/DeepStore/data/small-scale/caida/raw.csv'
    # # Path to file containing paths of generated CSVs
    # gen_data_list_path = '/ocean/projects/cis230086p/sdong3/DeepStore/experiments/output/gen_caida.csv'

    # # Load the real dataset
    # real_df = pd.read_csv(csv_path_real)

    # # List to store (path, packet_avg, stateless_avg, stateful_avg)
    # avg_results = []

    # # Read each path and evaluate
    # with open(gen_data_list_path, 'r') as f:
    #     for line in f:
    #         gen_path = line.strip()
    #         if not gen_path or not os.path.isfile(gen_path):
    #             print(f"Skipped invalid path: {gen_path}")
    #             continue

    #         gen_df = pd.read_csv(gen_path)
    #         rst = eval_metrics(real_df, gen_df, n=10)

    #         avg_packet = rst["packet"]["avg"]
    #         avg_stateless = rst["flow_stateless"]["avg"]
    #         avg_stateful = rst["flow_stateful"]["avg"]

    #         avg_results.append((gen_path, avg_packet, avg_stateless, avg_stateful))

    #         print(f"Evaluated {gen_path}")
    #         # print(f"  Packet Avg:     {avg_packet:.4f}")
    #         # print(f"  Stateless Avg:  {avg_stateless:.4f}")
    #         print(f"  Stateful Avg:   {avg_stateful:.4f}")
    #         print("--------------------------------------------------")

    # # Summary
    # print("\n=== Summary of All Evaluations ===")
    # for row in avg_results:
    #     print(row)

