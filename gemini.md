# Gemini Project - Spiking Neural Network fMRI Classification

This project uses a spiking neural network to classify fMRI data.

## Key Technologies

*   Python
*   PyTorch
*   slayerSNN
*   nilearn
*   simtb

## Project Structure

*   `python/`: Contains the main Python source code for the project.
*   `matlab/`: Contains MATLAB scripts for data simulation and analysis.
*   `data/`: Contains the data used in the project.
*   `models/`: Contains the trained models.
*   `yamls/`: Contains the YAML configuration files.

## Common Tasks

### Install Dependencies

To install the required dependencies, run the following command:

```bash
pip install -r requirements.txt
```

### Run the main experiment

To run the main experiment, execute the following command:

```bash
python python/01_EXPERIMENT.py
```

### Run the simtb training

To run the simtb training, execute the following command:

```bash
python python/trn_simtb_dc.py
```
