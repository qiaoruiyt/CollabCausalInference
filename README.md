# Collaborative Causal Inference

This repository contains the code for the paper "Collaborative Causal Inference with Fair Incentives".

## Requirements

The main code is written for and tested on Python 3.8.11.
Python 2.7 may be required if you wish to process TCGA dataset.
The program requires scikit-learn, numpy, scipy, matplotlib, which can be installed from requirements.txt.

## Datasets

### IHDP

- The IHDP data is attached (source: <https://github.com/AMLab-Amsterdam/CEVAE>)

### JOBS

- Download JOBS data from <https://users.nber.org/~rdehejia/data/.nswdata2.html>

```bash
mkdir datasets/JOBS
cd datasets/JOBS
wget https://users.nber.org/~rdehejia/data/nsw_treated.txt 
wget http://www.nber.org/~rdehejia/data/nsw_control.txt 
```

### TCGA

- Install perfect_match from <https://github.com/d909b/perfect_match> (It is recommended to use Python 2 as required by perfect_match repository just to process this dataset)
- Download tcga.db from perfect_match
- Run the script data_processing.py 

```bash
conda create -n pmdata python=2.7
conda activate pmdata
git clone https://github.com/d909b/perfect_match
cd perfect_match 
pip install .
pip install Keras==2.0.0 
wget https://paperdatasets.s3.amazonaws.com/tcga.db 
cp ../data_processing.py ./data_processing.py
python data_processing.py
mv tcga_compact.txt ../datasets/TCGA
```
The perfect_match repo is no longer needed after the dataset is successfully processsed. 

## Experiments 

### Main

To experiment collaborative causal inference on datasets:
```bash
python main.py --dataset [dataset_name] --n_partitions [num_of_partitions]
```

By default, there are 5 equal-sized partitions. The time complexity is $O(N^2)$ with respect to the number of partitions due to Shapley value. 


