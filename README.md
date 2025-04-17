# Stochastic Submodular Data Forgetting
<img width="1014" alt="forgetting_round" src="https://github.com/user-attachments/assets/05d6373b-af1d-46c2-ac28-31bd838059c2" />

## Description
This repository contains the code and data of the paper Stochastic Submodular Data Forgetting.
## Requirements
```
pip install numpy
pip install scikit-learn
pip install pandas
pip install statistics
```
## Data Download
REAL DATASETS: The 3 real datatasets are hosted on Google Drive. You can download it using the following link: 
- [Download Data](https://drive.google.com/file/d/1YjCt-RZUyEHslqmA3yJHJi-Tk6SNFlbP/view?usp=sharing)

Once you download it please un-zip it and place it in the same directory as the source files.

SYNTHETIC DATASETS: To generate the 5 synthetic datasets please run:
```
python3 synthetic_data_generation.py
```
## General Usage
In order to run a data forgetting round execute:
```
python3 forgetting_round.py [alg_choice] [dataset_choice] [percentage_of_db] [percentage_of_ql] [budget] [n_iterations] [av_stdevs_calculation] [only_time]
```
The command line arguments correspond to:
- `[dataset_choice]`: The desired algorithm to run. Select from the list [`IndepDF`,`DepDF`,`LAZY GREEDY`,`LAZY GREEDY_FAST_NN`,`QUERY-BASED-AMNESIA`]
- `[dataset_choice]`: The desired dataset $D$ inside the `sample_data` folder. Select one from the list [flight, photo, wiki].
- `[percentage_of_db]`: The percentage of $D$ to use as input data. Select a number inside the continuous interval [0, 1].
- `[percentage_of_ql]`: The percentage of queries from the corresponding query-log $Q$ to use as input log. Select a number inside the continuous interval [0, 1].
- `[budget]`: The budget $B$ is the percentage of $D$ to be kept. Select a number inside the continuous interval [0,1].
- `[n_iterations]`: The number of gradient ascent iterations $T$ the `DepDF` routine will perform.
- `[av_stdevs_calculation]`: Whether the average answer set diversity is computed or not. Select `0` (False) not to perform the calculation, and `1` (True) to perform it.
- `[only_time]`: Whether $f(D*)$ is evaluated or not for solution $D*$ or only the time taken to build $D*$ is reported. Select `0` (False) to return both the time and function evaluation and `1` (True) to just return the time.

An example is given in the code snippet below,
```
python3 forgetting_round.py DepDF flight 1 0.25 0.01 2000 0 0
```
In this example, \texttt{DepDF} is run such that $D$ is the full flights dataset, $Q$ is the 25% of the query-log, $B$ is 1% of $|D|$, $T= 2000$, the average answer set diversity is not computed, and both the time taken to build $D*$ and $f(D*)$ are reported.
