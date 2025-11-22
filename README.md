# AINTEC Tutorial Instructions
This repository contains tutorials and experiments for generative models on network trace datasets. This instruction is for AINTEC'2025 tutorial, which is taylored to its specific enviroment and setup. For general purpose use case please refer to the `README.md` in `main` branch.

## Set up environement

> [!NOTE]
> **Hardware Requirements:** This `README.md` is for AINTEC'2025 tutorial ONLY.

**Step 1:** Open a terminal window and connect to COARE Saliksik HPC using the following accounts:
```bash
username: aintec-trainee-[01...40] # for trainees
password (all accounts): aintec@2025
```
On your terminal:
```bash
ssh <user>@saliksik.asti.dost.gov.ph
#then enter password
```

**Step 2:** Clone the repo to your server
```bash
git clone https://github.com/netsharecmu/generative-trace-tutorials.git
cd generative-trace-tutorials
```

**Step 3:** Use this salloc command to allocate a GPU node to your account:
```bash
salloc -p gpu_a100 -q 2c-1h_gpu-a100_1g.10g --gres=gpu:1 --reservation=aintec-workshop
```
Wait until a node is allocated to you.

**Step 4:** Use the module command to load the workshop environment:
```bash
module load aintec-2025
```

**Step 5:** Run this command to start the Jupyter Notebook:
```bash
srun run-notebook
```

**Step 6:** While the notebook is starting, open another terminal window run the ssh command printed by the previous command:
```bash
# look for the ssh command that looks like this and run it on another terminal window:
ssh -vv -NL <port>:<hostname>:<port> <user>@saliksik.asti.dost.gov.ph

# then enter the same password
aintec@2025
```

**Step 7:** Go back to the previous terminal window and use the localhost (127.0.0.1) link (the third link from the screenshot below) to connect to the notebook using a browser. You are now connected to COARE’s Saliksik HPC Cluster through Jupyter Notebook

## Run experiments
In this tutorial we have two datasets: 
- **Tabular Dataset**: `data/sample_tabular_data.csv`.
- **Network Dataset**: `data/caida-10k.csv`.

For each dataset, we have notebooks for training, evaluation and downstream tasks.

### Tabular Dataset
- **Training:** We have `tabular_CTGAN.ipynb` using CTGAN and `tabular_RealTabFormer.ipynb` using RealTabFormer.
- **Evaluation:** `tabular_quality_check.ipynb` evaluates the data quality using both average JSD and customized queries written by domain experts.
- **Downstream Task:** `tabular_tasks.ipynb` using synthetic data for data augmentation in ML predictor training. We use two different ML predictors (SVM and Dicision Tree) here.

### Network Dataset
- **Training:** We have `network_ctgan.ipynb` using CTGAN and `network_netshare.ipynb` using NetShare.
- **Evaluation:** `network_quality_check.ipynb` evaluates the data quality using both average JSD and customized queries written by domain experts.
- **Downstream Task:** `network_tasks.ipynb` using synthetic data for network measurement system testing. Specifically we test two measurement algorithms (SpaceSaving and Count-Min Sketch+Heap) on their hit rate for the Top-K most frequent flow identification.
