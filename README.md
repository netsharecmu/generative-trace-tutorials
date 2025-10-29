# Download datasets (small-scale)
Download from [Google Drive link](https://drive.google.com/drive/folders/15wt9iHmBbqZpSA6sLtpgQ_6DSsUDEChd?usp=drive_link) and put under `data/` folder.

# Install dependencies

## Step 1: Install uv (recommended)
```bash
# Install uv (fast Python package installer)
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Step 2: Create virtual environment and install dependencies
```bash
# Create a virtual environment with Python 3.9
uv venv --python 3.9

# Activate the virtual environment
source .venv/bin/activate  # On macOS/Linux
# or .venv\Scripts\activate on Windows

# Install all dependencies from requirements.txt
uv pip install -r requirements.txt
```

Note: D3VAE has special dependencies. Follow the [doc](requirements.txt).

### Alternative: Using conda (if preferred)
```bash
conda create --name generative-trace-tutorials python=3.9
conda activate generative-trace-tutorials
pip install -r requirements.txt
```

# Change the configuration
1. [small config file](config_small_scale_onboard.py)
    - `NETGPT_BASE_FOLDER = <your DeepStore working directory>`

# Run experiments
Example:
```bash
python3 driver.py \
    --config_partition config_small_scale_onboard \
    --dataset_name ${dataset_name} \
    --model_name ${model_name} \
    --order_csv_by_timestamp \
    --export_bert_embeddings
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
    --order_csv_by_timestamp \
    --export_bert_embeddings
```

# TODO
- [ ] Bump Python version to 3.12