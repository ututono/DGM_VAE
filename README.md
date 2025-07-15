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
│   │   ├── application.log
│   │   └── ...
│   └── latest/
│       ├── application.log
│       └── ...
├── outputs/
│   ├── 2025-06-24_16-00-00/
│   │   ├── artificats/
│   │   │   ├── generated_samples.png
│   │   │   └── ...
│   │   └── model/
│   │       ├── model.pth.tar
│   │       └── records.json
│   └── latest/
│       ├── artificats/
│       │   ├── generated_samples.png
│       │   └── ...
│       └── models/
│           ├── model.pth.tar
│           └── ...
├── core/
│   ├── agent.py
│   └── ...
├── README.md
├── training.py
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
### Training
To run the training script for vanilla VAE on MedMNIST dataset, use the following command:

```bash
python training.py \
    --dataset_names "ChestMNIST" \
    --model_name "vae" \
    --batch_size 64 \
    --epochs 10 \
    --learning_rate 0.001 \
    --image_size 28 \
    --save_model \

```

To run the training script for conditional VAE on MedMNIST dataset, use the following command:

```bash

python training.py \
    --dataset_names "ChestMNIST" \
    --model_name "cvae" \
    --batch_size 64 \
    --epochs 10 \
    --learning_rate 0.001 \
    --image_size 28 \
    --condition_dim 32 \
    --save_model 
```

Multiple datasets can be specified by separating them with commas, e.g., `--dataset_names "ChestMNIST,PathMNIST"`. 
You can also adjust the portions of different datasets by using the `--dataset_weights` flag, e.g., `--dataset_portions "0.5,0.5"` for equal portions of both datasets.

### Evaluation
To run the evaluation script on the trained model, use the following command:

```bash
python test.py \
    --checkpoint_path "outputs/latest/model" \
    --dataset_names "ChestMNIST" \
    --model_name "vae" \
    --image_size 28 \
    --batch_size 64
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

In order to facilitate the development and testing process, the repository includes a `smoke_test` flag that can be used to run a minimal set of tests locally. This is useful for quickly verifying that the code changes do not break the basic functionality of the training and evaluation scripts. Precisely, in this mode, the training script runs for:
- 10 images for training, 2 for val and 1 for test
- Epochs reduced to 1
- Batch size reduced to 2
- Image size reduced to 28x28
- Latent dimension reduced to 32
- Learning rate reduced to 0.001 (faster convergence)

To run the same tests locally that are run in the CI/CD pipeline, use the following commands:

```bash
# Run training test
python training.py --dataset_names pathmnist --image_size 28 --disable_mlflow --save_model --smoke_test

# Run evaluation test
python test.py --checkpoint_path outputs/latest/model --dataset_names pathmnist --image_size 28 --smoke_test
```

## Fine-tuning and Hyperparameter Optimization

Here is a spreadsheet where you can find the results of the hyperparameter optimization experiments conducted on the MedMNIST dataset: [Hyperparameter Optimization Results](https://docs.google.com/spreadsheets/d/1PHzUC_Qt4-nAHUrhF_N-p71zXtW7jW4IrrTwQswZ1i8/edit?usp=sharing).
