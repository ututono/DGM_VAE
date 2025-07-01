# DGM_VAE

This repository contains code for training and evaluating various types of Variational Autoencoders (VAEs) on the MedMNIST dataset.

## Setup
### Environment setup
Create a conda environment and install packages necessary for this project with the following command:

```bash
conda create --name dgm_vae python=3.10
conda activate dgm_vae
pip install -r requirements.txt
```
PS: `requirments_mini.txt` contains the minimal set of requirements to run the code, made by `pipreqs`, in order to avoid dependency conflicts with the `requirements.txt` file.

Create a `.env` file in the root directory of the project with the following content:

```text
PROJECT_ROOT=/PATH/TO/PROJECT
ENVIRONMENT=development # or production, used to set the logging level
```

### Output organization overview
```text
.
├── logs/
│   ├── 2025-06-24_16-00-00/
│   │   ├── app.log # std runtime log
│   │   └── error.log 
│   └── ...
├── mlruns
├── core
├── README.md
└── ...
```
runtime logs are stored in `logs/` directory, with a subdirectory for each run, named by the date and time of the run. The `mlruns` directory is used by [MLflow](https://mlflow.org/docs/latest/ml/) to store experiment runs and artifacts.

To view the MLflow UI, run the following command:

```bash
cd /PATH/TO/PROJECT/mlruns
mlflow ui --port 15000
```

Then open your browser and go to `http://localhost:15000`.

If you don't want to use MLflow, you can disable it by passing the `--disable_mlflow` flag when running the training script. The artifacts and metrics will be saved in the `outputs` directory instead.

## Usage Example
To run the training script for vanilla VAE on MedMNIST dataset, use the following command:

```bash
python training.py \
    --dataset_name "ChestMNIST" \
    --model_name "vanilla_vae" \
    --batch_size 64 \
    --epochs 10 \
    --learning_rate 0.001 \
    --image_size 28 \
    
```