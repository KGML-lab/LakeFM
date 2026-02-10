# ⚙️ Environment Setup

### 1. Create a new conda environment

```bash
conda create -n lakefm python=3.11
```
### 2. Activate the environment

```bash
conda activate lakefm
```
### 3. Install dependencies

Make sure you have the `requirements.txt` file available in the project directory.

Then install all required packages using `pip`:

```bash
pip install -r requirements.txt
```
### 4. (Optional) Verify installation

You can check that all necessary packages are installed:

```bash
pip list
```

### 5. Install PyTorch separately

You must install PyTorch separately according to your **CUDA version**.
Refer to the official PyTorch guide:

👉 https://pytorch.org/get-started/previous-versions/



### Example install command for CUDA 11.8:

```bash
pip install torch==2.6.0+cu118 torchvision==0.21.0+cu118 torchaudio==2.6.0 --extra-index-url https://download.pytorch.org/whl/cu118
```

## Datasets

For using any of the data, add the dataset to the corresponding dir (mentioned for each of the dataset below) under `resources`. 

1. FCR Simulation dataset [Download](https://drive.google.com/file/d/19kR_CIA3Z-bmx1I7ef5ziFNG2bgEuPAH/view?usp=drive_link) (`/resources/lakefm/data/FCR_data`)

2. WQHanson Simulation dataset [Download](https://drive.google.com/file/d/1GKB0xqkmKJHCIWBSPoP5hBFhE-IOW9cH/view?usp=drive_link) (`/resources/lakefm/data/WQHanson_Simulation`)

3. LakeBeD dataset [Download](https://drive.google.com/file/d/10OzRxqh0RIrM7XMFY7wvlYT36uIGMjnZ/view?usp=drive_link) (`/resources/lakefm/data/LakeBeD-US`)

---
## Running the code

Navigate to the `src/` directory:

```bash
cd src
```

Then, run the pretraining script:

```bash
bash scripts/pretrain.sh
```

The `scripts/pretrain.sh` script internally runs:

```bash
export CUDA_VISIBLE_DEVICES=0

torchrun --nproc_per_node=1 -m cli.train \
    -cp conf/pretrain \
    trainer.max_epochs=100 \
    trainer.wandb_name="fastdev_singleds_add" \
    model.tokenization="scalar" \
    dataloader.batch_size=8 \
    model.add_or_concat="add"
```
`n_proc_per_node` represents the number of GPUs to use <br>
`wandb_name` corresponds to the wandb run name <br>
**Please specify (i.e. you will create one when you start running the code) the wandb project name as,** <br>
- `trainer.wandb_project=<whatever_name_you_want_to_give>` <br>


_For different datasets we have different scripts_

You can modify parameters like `trainer.max_epochs`, `dataloader.batch_size`, `model.tokenization`, and others either directly in this script or by overriding them on the command line.
<br>
Basically, when we run this script, train.py automatically loads the default.yaml file located in `conf/pretrain/default.yaml` (thanks to Hydra). Here's a brief overview of how hydra works
- At the start of the process, hydra looks for this default.yaml and loads it as a dictionary (sort of). The yaml file can be seen as a tree/graph structure with each field as a node and anything inside as child nodes. So, for example if we have a root node as "dataloader" and want to change a parameter (or child node) inside it, we can just specify it as `dataloader.batch_size=32` <br>
- Plus, within default.yaml if we mention different yaml files, they will be loaded as well by hydra in the form of children dictionaries. In our default.yaml, we have for model related details, a reference to model.yaml (present as `base_model.yaml`), and for data details, a data.yaml (present as `base.yaml`)
---

## YAML File Structure

- `cli/conf/pretrain/default.yaml`  
  ➔ Base configuration that links data, model, and trainer YAMLs. <br>
  ➔ Contain `trainer` and `dataloader` details - so anything related to training and dataloader details - just check here <br>
  ➔ Also, sequence length related parameters (`seq_len` or past window and `pred_len` or future window) are present as root nodes here <br>

- `cli/conf/pretrain/data/<dataset>.yaml`  
  ➔ Dataset-specific configuration (paths, preprocessing settings).  <br>
  ➔ Modify this to change dataset related details <br>
  ➔ `FCR_Simulation_Phy.yaml` corresponds to FCR Simulation datasets <br>
  ➔ `WQ_Hanson_Simulation.yaml` corresponds to Paul Hanson's Simulation datasets <br>

- `cli/conf/pretrain/model/<model>.yaml`  
  ➔ Model architecture settings (e.g., `embedding dimensions`, `number of layers`, `vocabulary size`, etc).  <br>
  ➔ Useful parameters to change for model ablations <br>
  - `add_or_concat`: addition or concatenation of embedding information
  - `tokenization`: type of tokenization to apply - `patch` or `temporal` (not implemented yet) or `scalar` (default) <br>
  ➔ Present as `base_model.yaml`

---

## Evaluation Process

### Evaluation Scripts

Following scripts exist to different steps of evaluation:
1. eval.sh --> basic evaluation for a specific model on a specific dataset/s
2. eval_all.sh
  This script automates the evaluation of a pretrained LakeFM model across multiple lakes using the base `torchrun` eval command. THis script will loops over all lake IDs (e.g., 1–3 for hanson, 1–21 for lakebed) and save evaluation outputs (metrics, predictions, logs) in per-lake subfolders taht can be later used for analysis.
  ---

  #### **Usage**
  ```bash
  ./scripts/eval_all.sh <dataset> <ckpt_folder_name> [pretrain_datasets]
  ```
  
  - dataset: Test Dataset name — either hanson or lakebed
  - ckpt_folder_name: Folder name of the pretrained model checkpoint
  - pretrain_datasets (optional): Comma-separated list of datasets used for pretraining e.g., FCR_Simulation_Phy,WQ_Hanson_Simulation,LakeBeD
  - Results are saved to :
  ```bash
  <root_path>/lakefm/dev/experiments/evaluations_pretrain/<ckpt_folder>/<dataset>/lake<ID>/
```
---

3. aggregate_evals.py

Aggregates per-lake `evaluation_summary.json` files into a single CSV for easy comparison.  
Supports both lake-level and variable-level metrics from structured evaluation runs.

**Input**:  
- `run_folder`: Path to folder like `evaluations_pretrain/<ckpt_folder>`  
- `--csv` (optional): Output CSV path (appended if exists)

**Output**:  
- A CSV file summarizing lake-wise and variate-wise metrics per run

**Example**:
```bash
python scripts/aggregate_eval_jsons.py evaluations_pretrain/hanson_pretrain_scalar_concat --csv hanson_results.csv
```
4. eval_all_caller.sh --> calls eval_all.sh and aggregate_evals.py, should be customised for specific runs, analysis and use case



