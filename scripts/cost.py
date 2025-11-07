#!/usr/bin/env python3
"""
Interactive cost estimator for storing raw traces vs. DGM (Deep Generative Model) artifacts.

On start, the program asks the user for:
  - GPU cost per hour (default: 0.20 $/hour)
  - Storage cost per GB per year (default: 0.24 $/GB-year)
  - Raw trace size in GB (required)
  - Model training time in seconds (required)
  - Model storage size in GB (required)

Outputs:
  1) Raw trace retention cost per year
  2) DGM cost for year 1 (one-year model retention + training cost)
  3) Break-even years of retention where cumulative DGM cost becomes cheaper than cumulative raw retention

Break-even condition:
  Find Y such that:  Y * model_year_cost + training_cost <= Y * raw_year_cost
  If model_year_cost >= raw_year_cost and training_cost > 0, break-even is never.
"""
from __future__ import annotations
import math
from dataclasses import dataclass

# ---------- Data models ----------
@dataclass
class Inputs:
    raw_size_gb: float
    model_size_gb: float
    training_time_sec: float
    storage_rate_gb_year: float  # $ per GB-year (applied to both raw & model)
    gpu_rate_per_hour: float     # $ per training hour

    @property
    def yearly_raw_cost(self) -> float:
        return self.raw_size_gb * self.storage_rate_gb_year

    @property
    def yearly_model_cost(self) -> float:
        return self.model_size_gb * self.storage_rate_gb_year

    @property
    def training_cost(self) -> float:
        return (self.training_time_sec / 3600.0) * self.gpu_rate_per_hour

@dataclass
class Results:
    yearly_raw_cost: float
    yearly_model_cost: float
    training_cost: float
    dgm_year1_cost: float
    breakeven_years_exact: float | None
    breakeven_years_ceiling: int | None

# ---------- Core computation ----------

def compute(inputs: Inputs) -> Results:
    yr_raw = inputs.yearly_raw_cost
    yr_model = inputs.yearly_model_cost
    train_cost = inputs.training_cost
    dgm_year1 = yr_model + train_cost

    # Break-even: Y >= train_cost / (yr_raw - yr_model)
    delta = yr_raw - yr_model
    if delta <= 0:
        be_exact = None
        be_ceil = None
    elif train_cost <= 0:
        be_exact = 0.0
        be_ceil = 0
    else:
        be_exact = train_cost / delta
        be_ceil = math.ceil(be_exact)

    return Results(
        yearly_raw_cost=yr_raw,
        yearly_model_cost=yr_model,
        training_cost=train_cost,
        dgm_year1_cost=dgm_year1,
        breakeven_years_exact=be_exact,
        breakeven_years_ceiling=be_ceil,
    )

# ---------- Helpers ----------

def fmt_usd(x: float | None) -> str:
    if x is None:
        return "—"
    return f"${x:,.2f}"


def prompt_float(prompt: str, *, default: float | None = None, required: bool = False) -> float:
    while True:
        raw = input(prompt).strip()
        if not raw:
            if default is not None:
                return float(default)
            if not required:
                return 0.0
        try:
            return float(raw)
        except ValueError:
            print("Please enter a valid number.")

# ---------- Main (interactive) ----------

def main() -> None:
    print("DeepStore Cost Estimator\n")

    gpu_rate = prompt_float("GPU cost per hour [$] (default 0.20): ", default=0.20)
    storage_rate_year = prompt_float(
        "Storage cost per GB per year [$] (default 0.24): ", default=0.24
    )

    raw_gb = prompt_float("Raw trace size [GB]: ", required=True)
    train_sec = prompt_float("Model training time [seconds]: ", required=True)
    model_gb = prompt_float("Model storage size [GB]: ", required=True)

    inputs = Inputs(
        raw_size_gb=raw_gb,
        model_size_gb=model_gb,
        training_time_sec=train_sec,
        storage_rate_gb_year=storage_rate_year,
        gpu_rate_per_hour=gpu_rate,
    )

    res = compute(inputs)

    print("\n=== Cost Summary ===")
    print(f"Raw size:                 {raw_gb:,.2f} GB")
    print(f"Model size:               {model_gb:,.2f} GB")
    print(f"Storage rate:             ${storage_rate_year:.4f}/GB-year")
    print(f"Training time:            {train_sec:,.2f} sec  (~{train_sec/3600.0:.2f} h) @ ${gpu_rate:.2f}/h")

    print("\n--- Annualized Costs ---")
    print(f"Raw retention (1y):       {fmt_usd(res.yearly_raw_cost)}")
    print(f"Model retention (1y):     {fmt_usd(res.yearly_model_cost)}")
    print(f"Training (one-time):      {fmt_usd(res.training_cost)}")

    print("\n--- DGM vs Raw ---")
    print(f"DGM cost in year 1:       {fmt_usd(res.dgm_year1_cost)}  (model 1y + training)")

    if res.breakeven_years_exact is None:
        if res.yearly_model_cost >= res.yearly_raw_cost and res.training_cost > 0:
            print("Break-even years:         never (yearly model retention ≥ yearly raw retention)")
        else:
            print("Break-even years:         not applicable")
    else:
        print(f"Break-even years (exact): {res.breakeven_years_exact:.3f} years")
        print(f"Break-even (ceiled):      {res.breakeven_years_ceiling} years (first whole year where DGM is cheaper)")

    print()  # trailing newline


if __name__ == "__main__":
    main()