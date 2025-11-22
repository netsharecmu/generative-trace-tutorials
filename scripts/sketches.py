import mmh3
import heapq
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Hashable, Iterable, List, Optional, Tuple


@dataclass
class _Entry:
    key: Hashable
    count: int
    error: int  # lower bound on how much we may be overcounting


class SpaceSavingSketch:
    """
    Space-Saving sketch for tracking frequent items with fixed memory.

    - capacity: max number of counters we keep.
    - update(key, weight): process a new occurrence of `key`.
    - estimate(key): return the estimated frequency (over-estimate).
    - topk(k): return the k most frequent items (by estimated count).
    - heavy_hitters(min_count): all items with estimated count >= min_count.

    Complexity:
        update:  O(capacity) (due to linear scan for min counter)
        query:   O(1)
        topk:    O(capacity log capacity)
    """

    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._table: Dict[Hashable, _Entry] = {}

    # -----------------------
    # Core update operations
    # -----------------------
    def update(self, key: Hashable, weight: int = 1) -> None:
        """
        Process an item with a given weight (default 1).
        """
        if weight <= 0:
            return  # ignore non-positive updates

        # Case 1: key already tracked
        if key in self._table:
            entry = self._table[key]
            entry.count += weight
            return

        # Case 2: we still have room, add new entry with zero error
        if len(self._table) < self.capacity:
            self._table[key] = _Entry(key=key, count=weight, error=0)
            return

        # Case 3: table full -> replace the smallest counter
        # Find entry with minimum count
        min_key, min_entry = min(self._table.items(), key=lambda kv: kv[1].count)

        # Replace it with the new key
        # Space-Saving rule:
        #   new_count  = min_count + weight
        #   new_error  = min_count
        min_count = min_entry.count
        self._table.pop(min_key)

        self._table[key] = _Entry(
            key=key,
            count=min_count + weight,
            error=min_count,
        )

    def bulk_update(self, keys: Iterable[Hashable]) -> None:
        """
        Convenience method: update sketch with a stream of keys (weight=1 each).
        """
        for k in keys:
            self.update(k)

    # -----------------------
    # Query methods
    # -----------------------
    def estimate(self, key: Hashable) -> int:
        """
        Return the estimated count for `key`.

        This is an upper bound on the true frequency:
            true_count(key) <= estimate(key)
            true_count(key) >= estimate(key) - error(key), if key is present
        If `key` is not tracked, returns 0.
        """
        entry = self._table.get(key)
        return entry.count if entry is not None else 0

    def estimate_with_error(self, key: Hashable) -> Tuple[int, int]:
        """
        Return (estimate, error) for `key`.
        If key is not tracked, returns (0, 0).
        """
        entry = self._table.get(key)
        if entry is None:
            return 0, 0
        return entry.count, entry.error

    def topk(
        self, k: Optional[int] = None
    ) -> List[Tuple[Hashable, int, int]]:
        """
        Return the top-k items as a list of (key, estimate, error),
        sorted by estimated count descending.

        If k is None or k >= capacity, returns all tracked items.
        """
        if k is None or k >= len(self._table):
            k = len(self._table)

        entries = sorted(
            self._table.values(), key=lambda e: e.count, reverse=True
        )
        entries = entries[:k]
        return [(e.key, e.count, e.error) for e in entries]

    def heavy_hitters(
        self, min_count: int
    ) -> List[Tuple[Hashable, int, int]]:
        """
        Return all items with estimated count >= min_count, as
        (key, estimate, error), sorted by estimated count descending.
        """
        entries = [
            e for e in self._table.values() if e.count >= min_count
        ]
        entries.sort(key=lambda e: e.count, reverse=True)
        return [(e.key, e.count, e.error) for e in entries]

    # -----------------------
    # Utility / debugging
    # -----------------------
    def __len__(self) -> int:
        return len(self._table)

    def __contains__(self, key: Hashable) -> bool:
        return key in self._table

    def items(self) -> List[Tuple[Hashable, int, int]]:
        """
        Return all tracked items as (key, estimate, error).
        """
        return [(e.key, e.count, e.error) for e in self._table.values()]


class CountMinSketch:
    """
    Standard Count-Min Sketch with d hash functions and width w.
    """

    def __init__(self, width: int, depth: int, seed: int = 1234):
        self.width = width
        self.depth = depth
        self.seed = seed
        self.table = [[0] * width for _ in range(depth)]

    def update(self, key, weight=1):
        for i in range(self.depth):
            h = mmh3.hash(str(key), self.seed + i) % self.width
            self.table[i][h] += weight

    def estimate(self, key) -> int:
        """Return the CMS upper-bound estimate for the key."""
        est = float("inf")
        for i in range(self.depth):
            h = mmh3.hash(str(key), self.seed + i) % self.width
            est = min(est, self.table[i][h])
        return est


import heapq

class CountMinHeap:
    """
    Count-Min Heap: top-K heavy hitters using:
        - A Count-Min Sketch (CMS)
        - A size-K min-heap tracking high-frequency items

    API:
        update(key, weight=1)
        estimate(key)
        topk()  -> [(est, key), ...] sorted by est desc
    """

    def __init__(self, k: int, width: int = 2000, depth: int = 5):
        self.k = k
        self.cms = CountMinSketch(width=width, depth=depth)
        self.heap = []          # stores (est, key)
        self.entry_finder = {}  # key -> (est, key)

    # ---------------- internal helpers ----------------
    def _clean_heap_min(self):
        """
        Ensure heap[0] is a valid, up-to-date entry.
        Pops stale entries (those not matching entry_finder).
        Returns (est, key) or (None, None) if heap empty.
        """
        while self.heap:
            est, key = self.heap[0]
            cur = self.entry_finder.get(key)
            if cur is None or cur[0] != est:
                # stale entry, drop it
                heapq.heappop(self.heap)
                continue
            return est, key
        return None, None

    # ---------------- core API ----------------
    def update(self, key, weight: int = 1):
        if weight <= 0:
            return

        # 1) Update the Count-Min Sketch
        self.cms.update(key, weight)
        est = self.cms.estimate(key)

        # 2) If already tracked, just update mapping and push new entry
        if key in self.entry_finder:
            self.entry_finder[key] = (est, key)
            heapq.heappush(self.heap, (est, key))
            return

        # 3) If we still have room in heap, insert directly
        if len(self.entry_finder) < self.k:
            self.entry_finder[key] = (est, key)
            heapq.heappush(self.heap, (est, key))
            return

        # 4) Otherwise, compare against current (cleaned) min
        min_est, min_key = self._clean_heap_min()

        # If heap was all stale for some reason, just insert
        if min_key is None:
            self.entry_finder[key] = (est, key)
            heapq.heappush(self.heap, (est, key))
            return

        if est > min_est:
            # Evict the true current min
            del self.entry_finder[min_key]
            heapq.heappop(self.heap)

            # Insert new key
            self.entry_finder[key] = (est, key)
            heapq.heappush(self.heap, (est, key))

    def estimate(self, key):
        return self.cms.estimate(key)

    def topk(self):
        """
        Return current top-K as a list of (est, key),
        sorted in descending order of est.
        """
        # Filter to only valid (non-stale) entries
        valid = []
        for est, key in self.heap:
            cur = self.entry_finder.get(key)
            if cur is not None and cur[0] == est:
                valid.append((est, key))

        # Sort and keep best k
        valid.sort(reverse=True)
        return valid[:self.k]
    
      
def evaluate_sketches_topk(
    df: pd.DataFrame,
    k: int,
    ss_capacity: int = None,
    cm_k: int = None,
    cm_width: int = 2000,
    cm_depth: int = 5,
):
    """
    Evaluate top-k identification hit rate for SpaceSavingSketch and CountMinHeap.

    Hit Rate = |TrueTopK ∩ SketchTopK| / k
    """

    required_cols = ['srcip', 'dstip', 'srcport', 'dstport', 'proto']
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")

    # -------- 1. True Top-K using exact counting --------
    grouped = (
        df.groupby(required_cols)
          .size()
          .reset_index(name="count")
          .sort_values("count", ascending=False)
    )

    true_topk = grouped.head(k)
    true_topk_keys = set(
        (row.srcip, row.dstip, row.srcport, row.dstport, row.proto)
        for _, row in true_topk.iterrows()
    )

    # -------- 2. Initialize sketches --------
    if ss_capacity is None:
        ss_capacity = 2 * k
    if cm_k is None:
        cm_k = k

    ss = SpaceSavingSketch(capacity=ss_capacity)
    cmh = CountMinHeap(k=cm_k, width=cm_width, depth=cm_depth)

    # -------- 3. Stream data into sketches --------
    for _, row in df[required_cols].iterrows():
        key = (row.srcip, row.dstip, row.srcport, row.dstport, row.proto)
        ss.update(key)
        cmh.update(key)

    # -------- 4. Extract sketch top-k --------
    ss_pred = ss.topk(k)
    ss_pred_keys = set(key for key, est, err in ss_pred)

    cm_pred = cmh.topk()[:k]
    cm_pred_keys = set(key for est, key in cm_pred)

    # -------- 5. Compute hit rates --------
    ss_hits = len(true_topk_keys & ss_pred_keys)
    cm_hits = len(true_topk_keys & cm_pred_keys)

    ss_hit_rate = ss_hits / k
    cm_hit_rate = cm_hits / k

    # -------- 6. Print and return --------
    print(f"SpaceSaving Hit Rate:  {ss_hit_rate:.4f}  ({ss_hits}/{k})")
    print(f"CountMinHeap Hit Rate: {cm_hit_rate:.4f}  ({cm_hits}/{k})")

    return {
        "ss_hit_rate": ss_hit_rate,
        "cm_hit_rate": cm_hit_rate,
        "ss_hits": ss_hits,
        "cm_hits": cm_hits,
        "true_topk_keys": true_topk_keys,
    }