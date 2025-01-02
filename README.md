# Surrogate Model Training and Analysis for Multi-Objective Optimization

This repository contains scripts and utilities to:
1. **Generate synthetic datasets** of original objective functions.
2. **Train surrogate models** on those datasets.
3. **Perform multi-objective optimization** using the trained surrogates.
4. **Extract Exploratory Landscape Analysis (ELA) features** using R and flacco.
5. (TODO) **Evaluate** the quality of the generated Pareto fronts and create a classifier for selecting the best surrogate approach.

---

## Repository Structure

Below is an overview of the key files and their roles:

### 1. Data Generation Scripts

- **`generate_dtlz.py`**, **`generate_WFG.py`**, **`generate_engineering.py`**,  
  **`generate_VehicleCrashworthiness.py`**, **`generate_RiverPollution.py`**,  
  **`generate_multiple_clutch_brakes.py`**  
  - Each script generates data for a specific multi-objective test problem (e.g., DTLZ, WFG, engineering design problems, etc.).  
  - The scripts produce CSV or similar outputs containing design variables (`x`) and corresponding objective values (`f`).  
  - This data is used later to train surrogate models.

- **`Create_dtlz_and_wfg_pareto.py`**, **`Create_engineering_pareto.py`**  
  - Generate or refine Pareto sets for the DTLZ/WFG and engineering test problems.

- **`Check_domination.py`**  
  - A utility for checking domination relationships among solution points of generated approximated Pareto fronts.  

### 2. Surrogate Model Scripts

- **`dtlz_surrogates.py`**, **`wfg_surrogates.py`**, **`engineering_surrogates.py`**,  
  **`vehicle_crashworthiness_surrogates.py`**, **`river_pollution_surrogates.py`**,  
  **`multiple_clutch_brakes_surrogates.py`**  
  - Train surrogate models (e.g., regression, Gaussian Processes, etc.) on the generated data.  
  - After training, perform multi-objective optimization using these surrogates to produce approximate Pareto fronts.  
  - The resulting fronts are stored for further evaluation.

### 3. Feature Extraction and ELA Scripts

- **`features.R`**, **`features_updated.R`**  
  - Use [FLACCO](https://cran.r-project.org/package=flacco) to compute Exploratory Landscape Analysis (ELA) features for the original problems.  
  - These features are saved into a dataset.

### 4. Job Submission Script

- **`puhtiJob.sh`**  
  - An example of a job script for CSC’s Puhti supercomputer.
  - It demonstrates how to automate running the surrogate scripts, feature extraction, and data generation in an HPC environment.

### 5. Miscellaneous

- **`.DS_Store`**  
  - A macOS system file, can generally be ignored.

- **`README.md`**  
  - This file you are reading now.

---

## Workflow Summary

1. **Data Generation**:  
   Run any of the `generate_*.py` scripts to produce datasets of (X, F) for the original multi-objective problems.

2. **Train Surrogates & Optimize**:  
   Use the `*_surrogates.py` scripts (e.g., `dtlz_surrogates.py`) to:
   - Train surrogate models on the generated datasets.
   - Perform multi-objective optimization using NSGA-III, RVEA and IBEA to trained surrogate models.
   - Store the resulting approximate Pareto fronts.

3. **Feature Extraction**:  
   Use `features_updated.R`) to:
   - Load the original problem’s data.
   - Compute ELA features with flacco.
   - Save these features.

4. **Evaluate Pareto Front Quality** (Todo):
   - **Indicator Computation**: Implement or integrate performance indicators to evaluate the quality of each generated Pareto front.
   - **Classification**: Build a classifier that takes the problem’s ELA features and indicator values as input and predicts which surrogate technique is best suited for that problem.

---

## Future Plans

- **Indicators**: Implement scripts or modules to assess the generated Pareto fronts with metrics such as hypervolume or IGD.  
- **Classifier**: Train a machine learning classifier to correlate ELA features with the best-performing surrogate approach.  

---
