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
    --save_model \

```

## CI/CD Pipeline

This project includes a GitHub Actions CI/CD pipeline that automatically tests the training and evaluation scripts when changes are pushed to the repository. The pipeline performs the following steps:

1. Sets up a Python 3.10 environment
2. Installs all required dependencies
3. Runs a basic training test with minimal epochs (2) to verify functionality
4. Runs the evaluation script on the trained model
5. Uploads training artifacts for inspection
6. Cleans up the environment after the tests are complete

The CI/CD pipeline is triggered on:
- Push to the dev branch
- Pull request to the dev branch
- [ ]  (TODO)dev as so far, should be changed to main in the future)

To view the status of the CI/CD pipeline, check the "Actions" tab in the GitHub repository.

### Running Tests Locally

To run the same tests locally that are run in the CI/CD pipeline, use the following commands:

```bash
# Run training test
python training.py --epochs 1 --batch-size 16 --dataset-name pathmnist --image-size 24 --latent-dim 8 --disable-mlflow --save-model

# Run evaluation test
python test.py --checkpoint-path outputs/latest/model/model.pt --dataset-name pathmnist --image-size 24 --latent-dim 8
```

Note: The CI/CD pipeline uses minimal resources (reduced epochs, batch size, image size, and latent dimension) to ensure efficient execution in the GitHub Actions environment.
