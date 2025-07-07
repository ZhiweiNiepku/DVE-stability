# DVE-stability
The official code of ["Predicting protein stability changes upon mutations with dual-view ensemble learning from single sequence"]. 


## Table of Contents
- [Overview](#overview)
- [Get Started](#get-started)
- [Data](#data)
- [Model training](#model-training)
- [Model inference](#model-inference)
- [License](#license)

## Overview
Predicting the protein stability changes upon mutations is one of the effective ways to improve the efficiency of protein engineering.
Here, we propose a dual-view ensemble learning-based framework, DVE-stability, for mutation-induced protein stability change prediction from single sequence.
DVE-stability integrates the global and local dependencies of mutations to capture the intramolecular interactions from two views through ensemble learning, in which a structural microenvironment simulation module is designed to indirectly introduce the information of structural microenvironment at the sequence level.
DVE-stability achieved state-of-the-art prediction performance on 7 single-point mutation benchmark datasets, and comprehensively surpassed other methods on 5 of them.
Furthermore, DVE-stability outperformed other methods comprehensively through zero-shot inference on multiple-point mutation prediction task, demonstrating superior model generalizability to capture the epistasis of multiple-point mutations.
More importantly, DVE-stability exhibited superior generalization performance in predicting rare beneficial mutations that are crucial for practical protein directed evolution scenarios.
In addition, DVE-stability identified important intramolecular interactions via attention scores, demonstrating interpretable.
Overall, DVE-stability provides a flexible and efficient tool for mutation-induced protein stability change prediction in an interpretable ensemble learning manner.


## Hardware requirements
The experiments are tested on one Tesla V100 (32GB).


## Get Started
Build the environment.
```python
pip install -r requirements.txt
```

## Data
Put the training and testing datasets in the "data" folder.
Download high-throughput fitness dataset at [mega-scale repo](https://zenodo.org/records/7401275).
Download C2878 and T2837 dataset at [Stability-oracle repo](https://github.com/danny305/StabilityOracle/tree/master/data/datasets).
Download S8754, M1261, S461 and S783 dataset at [GeoStab repo](https://github.com/Gonglab-THU/GeoStab/tree/main/data/ddG).



## Model training
To train this model on downstream tasks from scratch, please run
```python
python train.py
```

## Model inference

To test this model on different test datas, please run
```python
python test.py
```

## License
This project is licensed under the [MIT License](LICENSE).
