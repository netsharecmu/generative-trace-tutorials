# Generative Trace Tutorials
This repository contains tutorials and experiments for generative models on network trace datasets.

## Getting Started

> [!NOTE]
> **Hardware Requirements:** This project is currently tested and supported only on Linux (x86_64). M series Mac is not supported yet.

**Step 1:** Download datasets from [Google Drive link](https://drive.google.com/drive/folders/1l82vGWHRUhP-MM7ByauXYdo7pilk14IK?usp=drive_link) and put under `data/` folder.

**Step 2:** Install dependencies using conda:
```bash
conda create --name generative-trace-tutorials python=3.9
conda activate generative-trace-tutorials
pip install -r requirements.txt
```

Note: D3VAE has special dependencies. Follow the [doc](requirements.txt).

## Run experiments (Example)
```bash
python3 driver.py \
    --config_partition <config_partition> \
    --dataset_name ${dataset_name} \
    --model_name ${model_name} \
    --order_csv_by_timestamp
```
- `config_partition`: `small-scale` | `large-scale` | ... (see [config_small_scale.py](scripts/config_small_scale.py) for more details)
- `dataset_name`: `caida | dc | ca | m57 | ugr16 | cidds | ton`
- `model name`: 
    - `realtabformer-tabular`
    - `realtabformer-timeseries`
    - `ctgan`
    - `tabddpm`
    - `crossformer`
    - `d3vae`
    - `scinet`
    - `dlinear`
    - `patchtst`

A quick example:
```bash
python3 driver.py \
    --config_partition small-scale \
    --dataset_name caida \
    --model_name realtabformer-tabular \
    --order_csv_by_timestamp
```

# TODO
- [ ] Bump Python version to 3.12
- [ ] Add support for M series Mac