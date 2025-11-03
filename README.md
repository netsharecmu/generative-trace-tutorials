# Generative Trace Tutorials
This repository contains tutorials and experiments for generative models on network trace datasets.

## Hardware Requirements
**Note:** This project is currently tested and supported only on Linux (x86_64). M series Mac is not supported yet.

## Download datasets (small-scale)
Download from [Google Drive link](https://drive.google.com/drive/folders/1l82vGWHRUhP-MM7ByauXYdo7pilk14IK?usp=drive_link) and put under `data/` folder.

## Install dependencies

```bash
conda create --name generative-trace-tutorials python=3.9
conda activate generative-trace-tutorials
pip install -r requirements.txt
```

Note: D3VAE has special dependencies. Follow the [doc](requirements.txt).

## Run experiments
Example:
```bash
python3 driver.py \
    --config_partition config_small_scale_onboard \
    --dataset_name ${dataset_name} \
    --model_name ${model_name} \
    --order_csv_by_timestamp
```
- `config_partition`: you can change to your own config file
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
    --config_partition small-scale-onboard \
    --dataset_name caida \
    --model_name realtabformer-tabular \
    --order_csv_by_timestamp
```

# TODO
- [ ] Bump Python version to 3.12
- [ ] Add support for M series Mac