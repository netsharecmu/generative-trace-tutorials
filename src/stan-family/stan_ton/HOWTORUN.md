# How to run

```python run.py```

1. host_ips_prefix should contains all the possible host ip prefix.
2. Please be sure that the `out_file_dir` for method `preprocess` is the same with `train_folder` for method `prepare_standata`
3. To create synthetic data based on a pecific ip, change line 105 `df_rev = ntt.rev_transfer(samples)` to `df_rev = ntt.rev_transfer(samples, this_ip=<ip>)`
4. If you want to rerun, please delete all the files in stan_data/netflow