# grid_disruption

## Experimental Data Analysis
Intended directory structure:
```
  ├── Code-for-Ying-et-al.-2023
  |    ├── extracted_all: experimental data converted into python-readable format.
  |    ├── Figure_1
  |    ├── Figure_2
  |    └── ...
  └── grid_disruption: this directory
```

### Data Preparation
0. Make sure MATLAB is installed (preferably 2021b).
1. Clone [`Code-for-Ying-et-al.-2023`](https://github.com/johnson-ying/Code-for-Ying-et-al.-2023) repository and place `data_converter_all.m` inside the `Figure_2` directory.
2. Clone [CMBHOME](https://github.com/hasselmonians/CMBHOME) directly under the main directory (`Code-for-Ying-et-al.-2023`).
3. Launch MATLAB and run the following commands from inside `Figure_2` to convert the data to python-readable format.
```
addpath ../session_data
addpath ../session_data2
addpath ../CMBHOME
import CMBHOME.*
data_converter_all
```

### Data Processing
Run `ying2023_ratemap.py` to calculate the ratemaps as well as different metrics (example shell script: `ying2023_ratemap.sh`).

The results are stored inside the `data` directory.

### Figure Production
`analysis_methode.ipynb`: Figures to help explain the method
`analysis_heatmap.ipynb`: Heatmap
`analysis_ecdf.ipynb`: Empirical Cumulative Distributions

### CNN analysis
Run `classifier.py` to train a convolutional neural network.

`analysis_cnn.ipynb` produces corresponding figures.

## RNN Perturbation Analysis
1. Train the RNN by running `main.py` (example shell script: `run.sh`).
2. Run `eval.py` to perform perturbations (example shell script: `eval.sh`).
3. `analysis_rnn.ipynb` produces corresponding figures.