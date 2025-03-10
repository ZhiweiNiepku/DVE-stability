# DVE-stability
The official code of ["Protein stability prediction with dual-view ensemble learning from single sequence"]. 


## Table of Contents
- [Overview](#overview)
- [Hardware requirements](#hardware-requirements)
- [Get Started](#get-started)
- [Data](#data)
- [Model inference](#model-inference)
- [Model training](#model-training)
- [License](#license)

## Overview
Predicting the protein stability changes upon mutations is one of the effective ways to improve the efficiency of protein engineering.
Here, we propose a dual-view ensemble learning-based framework, DVE-stability, for protein stability prediction from single sequence.
DVE-stability integrates the global and local dependencies of mutations to capture the intramolecular interactions from two views through ensemble learning, in which a structural microenvironment simulation module is designed to indirectly introduce the information of structural microenvironment at the sequence level.
DVE-stability not only achieves comparable prediction performance to state-of-the-art methods, but also achieves superior performance in predicting rare beneficial mutations that are critical for directed evolution of proteins.
More importantly, DVE-stability identifies important intramolecular interactions via attention scores, demonstrating interpretable.
Overall, DVE-stability provides a flexible and efficient tool for protein stability prediction in an interpretable ensemble learning manner.

## Hardware requirements

The experiments are tested on one Tesla V100 (32GB).

## Get Started
Build the environment.
```python
pip install -r requirements.txt
```

## Data
Training and testing data are in the "data" folder.

Download mega-scale dataset for training at [mega-scale dataset](https://zenodo.org/records/7401275)


## Model inference
Download the checkpoint of DVE-stability and modify the paths in the code.
| Content  | Link  |
| ----- | -----|
| Model Checkpoint | [link](https://figshare.com/ndownloader/files/46044900) |

To test DVE-stability on test dataset, please run
```python
python test.py
```
## Model training
To train DVE-stability on downstream tasks from scratch, please run
```python
python train.py
```
## License
This project is licensed under the [MIT License](LICENSE).
