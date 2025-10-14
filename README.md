# Federated-Bayesian-Optimization
## Introduction
This repository contains the experimental data and code corresponding to the Federated Bayesian Optimization method presented in the paper "An Intelligent Distributed Chemical Twin System for Collaborative Material Discovery".
- The data folder stores the simulated experimental data used in the paper.
- SingleBO.py implements the single-node Bayesian Optimization. The results of running this file will be saved in the SingleResult folder. Experimental settings can be modified by adjusting the parameters in the main function, with specific explanations as follows:
  - datafile: Path to the data file.
  - totrnd: Number of experimental rounds with different initialized datasets.
  - trials: Number of repeated experiments conducted for each initialized dataset.
  - init_datanum: Quantity of initialized data.
  - max_datanum: Maximum number of recommended attempts per experimental round (the experiment will end early if the optimal value is found).
- FedBOv6.py is the Federated Bayesian Optimization code used in the paper. Experimental results are saved in the FedResultv6 folder. Experimental settings can be adjusted by modifying the parameters in the main function, with specific explanations as follows:
  - client_idx: Index of the single-node data used, which is for loading the initialized data from the SingleResult folder.
  - trials: Number of repeated experiments for the fixed combination of initialized datasets.
  - filepath: Path to all data files.
  - input_datapath: Path to the fixed initialized data file.
  - maxrounds: Maximum number of recommended attempts per experimental round.（The default settings will execute directly until all feasible combinations have been recommended.）
- plot.py is used to draw intuitive diagrams of the optimization path, and the path to the result folder must be specified.

## Reprodcue SingleBO & FedBO
```bash
git clone https://github.com/pic-ai-robotic-chemistry/Federated-Bayesian-Optimization.git
cd Federated-Bayesian-Optimization
conda env create -f environment.yml
conda activate FedBO

python SingleBO.py
python FedBOv6.py
```
