# Neural Architecture Search for Transformer Architecture

This repository implements a **Neural Architecture Search (NAS)** for the **Transformer architecture**. The objective is to determine the optimal number of heads and layers for a Transformer model, balancing **time** and **perplexity** based on the number of entries available for processing.

## Project Structure
The repository contains the following files and directories:

- `dataset/`          - Contains the training data in various subdirectories.
- `config/`           - Configuration files for training and preprocessing.
- `model/`            - Implementation of various model components.
- `results/`          - Scripts for analyzing the results.
- `logs/`             - Directory where logs and checkpoints are saved.
- `train.py`          - Script for training the Transformer model.
- `train-rl.py`       - Script for training the reinforcement learning model.
- `preprocess.py`     - Data preprocessing module.
- `env.py`            - Custom environment for reinforcement learning.
- `config_reader.py`  - Reads and parses configuration files.
- `run.sh`            - Shell script to run the training multiple times.
- `test_run.py`       - Script for testing the training process.
- `requirements.txt`  - List of dependencies.

## Installation
### Clone the Repository
To clone the repository, run the following commands:
```bash
git clone https://github.com/harshit-sandilya/NAS.git
cd NAS
```

### Install the Required Dependencies
To install the required dependencies, run:
```bash
pip install -r requirements.txt
```

### Create the `logs` Directory
To create the `logs` directory, run:
```bash
mkdir logs
```

## Usage

### Training the Transformer Model
To train the Transformer model, execute:
```bash
python train.py
```

### Training the Reinforcement Learning Model
To train the reinforcement learning model, execute:
```bash
python train-rl.py <mode>
```
where `<mode>` can be either `1` or `2`.

### Running the Training Script Multiple Times
To run the training script multiple times, use:
```bash
bash run.sh
```

### Analyzing Results
To analyze the results from the log files generated, execute:
```bash
python results/analyze.py
```

## Configuration
The configuration files are located in the `config/` directory. The main configuration files are:
- **train.json**: Configuration for training the Transformer model.
- **preprocess.json**: Configuration for data preprocessing.
- **deepspeed.json**: Configuration for DeepSpeed optimization.

## Model Components
### Transformer
The Transformer model is implemented in `model/Transformer.py`. It includes the following components:
- **Decoder**: Defined in `Decoder.py`.
- **DecoderBlock**: Defined in `DecoderBlock.py`.
- **MultiHeadAttention**: Defined in `MultiHeadAttention.py`.
- **PositionalEncoding**: Defined in `PositionalEncoding.py`.
- **Loss**: Defined in `Loss.py`.
- **Normalizations**: Defined in `Normalizations.py`.

### Reinforcement Learning Environment
The custom environment for reinforcement learning is implemented in `env.py`.

## Results and Discussion
The Neural Architecture Search was conducted over 10 test iterations using DQN as the reinforcement learning agent. The logs generated during these experiments are stored in the `results/logs` directory. The analysis of these logs, performed using the `results/analysis.py` script, is organized into three folders:

- **aggregate**: Displays the fluctuation of parameters over steps as a whole.
- **frames**: Shows the fluctuation of parameters over steps, with each reset initiating a new sequence.
- **two**: Compares the first and last frames to provide an overview of the NAS process.

Unfortunately, the results of these experiments were inconclusive. Due to the resource-intensive nature of the computations, further investigations had to be postponed.

## Acknowledgements
This project utilizes the following libraries:
- **PyTorch**
- **PyTorch Lightning**
- **Stable Baselines3**
- **TensorBoard**
- **SentencePiece**
- **PyBind11**
- **DeepSpeed**

For more details, refer to the `requirements.txt` file.
