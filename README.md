# COSEE

This is the code for the paper “COSEE: Consistency-Oriented Signal-based Early Exiting based on Calibrated Sample Weighting Mechanism“.

## Installation

This repository is tested on Python 3.6.13, PyTorch 1.8.0, and Cuda 11.1. Using a conda environemnt is recommended, for example:

```
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch
```

After installing the required environment, clone this repository, and install the following requirements:

```
git clone https://github.com/He-Jianing/COSEE.git
cd COSEE
pip install -r ./requirements.txt
```

## Usage

There are two scripts in the `scripts` folder, which can be run from the repo root, e.g., `scripts/train.sh`.

#### train.sh

This is for fine-tuning COSEE models.

#### eval_signal.sh

This is for evaluating fine-tuned COSEE models, given a number of different early exiting thresholds.




