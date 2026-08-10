# Multi-Objective Feature Selection
### NSGA-II for Medical Tabular Data

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Algorithm-NSGA--II-14b8a6?style=flat-square" />
  <img src="https://img.shields.io/badge/Domain-Medical%20Tabular-f59e0b?style=flat-square" />
  <img src="https://img.shields.io/badge/Scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white" />
  <img src="https://img.shields.io/badge/License-MIT-6366f1?style=flat-square" />
</p>

> Applying NSGA-II multi-objective evolutionary optimization to feature selection on medical tabular data. Simultaneously optimizes two competing objectives - maximizing classifier accuracy and minimizing feature count - surfacing the full Pareto-optimal trade-off curve rather than collapsing it into a single metric.

---

## Table of Contents

- [The Problem](#the-problem)
- [Key Result](#key-result)
- [Why Multi-Objective](#why-multi-objective)
- [Algorithm: NSGA-II](#algorithm-nsga-ii)
- [Experimental Setup](#experimental-setup)
- [Results](#results)
- [Baseline Comparison](#baseline-comparison)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Research Context](#research-context)
- [Citation](#citation)

---

## The Problem

In medical ML, feature selection is not a single-objective problem. Fewer features means lower data collection cost, reduced patient burden, more interpretable models, and lower overfitting risk. More features means higher potential accuracy. The optimal trade-off is dataset-dependent and cannot be determined by a single scalar metric.

Standard approaches (RFE, importance-based selection, PCA) collapse this trade-off into a single number and return one solution. They do not tell you what accuracy you give up by removing two more features, or what you gain by adding three.

**NSGA-II surfaces the entire trade-off curve. Practitioners choose any point on the Pareto front based on their deployment constraints.**

---

## Key Result

> **9 features out of 30 (70% reduction) at 94.74% test accuracy** - matching the best full-feature baseline (Grid Search RandomForest, 30 features, 94.74%) while discarding 21 features entirely.

---

## Why Multi-Objective

Single-objective feature selection methods used as baselines:

| Method | Test Accuracy | Features Used | Trade-off Visible |
|---|---|---|---|
| Grid Search RF | 94.74% | 30 | No |
| RF Importance + RF | 94.74% | 30 | No |
| RFE + RandomForest | 94.74% | 7 | No |
| PCA + RandomForest | 93.86% | 10 components | No |
| Random Search RF | 91.23% | 30 | No |
| **NSGA-II (this work)** | **94.74%** | **9** | **Yes - full Pareto front** |

RFE reaches 7 features at the same accuracy, but returns a single solution with no information about what the curve looks like between 7 and 30 features. NSGA-II returns the full front - every Pareto-optimal (accuracy, feature count) pair discovered during evolution.

---

## Algorithm: NSGA-II

Non-dominated Sorting Genetic Algorithm II maintains a population of candidate (feature subset, classifier hyperparameter) pairs and evolves them across generations toward the Pareto front using four mechanisms:

**Non-dominated sorting** ranks solutions by Pareto dominance. A solution A dominates B if A is no worse than B on all objectives and strictly better on at least one. Solutions are sorted into fronts: Front 1 contains all non-dominated solutions, Front 2 contains solutions dominated only by Front 1, and so on.

**Crowding distance** preserves diversity along each front by measuring how isolated a solution is from its neighbors in objective space. Solutions in sparse regions are preferred during selection, preventing the population from collapsing to a single point on the front.

**Tournament selection** selects parents for the next generation. Candidates are compared first by front rank, then by crowding distance.

**Elitism** combines parents and offspring into a pool of size 2N, then selects the top N by rank and crowding distance. Best solutions are never lost.

```
Initialize population of (feature mask, RF hyperparameters)
Population size: 64, Generations: 30
         |
         v
+---------------------+
|  Evaluate           |   CV training accuracy + feature count
|  (two objectives)   |   RandomForest with evolved hyperparameters
+----------+----------+
           |
           v
+---------------------+
|  Non-dominated Sort |   Assign Pareto front rank to each individual
|  + Crowding Distance|   Measure isolation within each front
+----------+----------+
           |
           v
+---------------------+
|  Tournament Select  |   Rank first, crowding distance as tiebreaker
|  Crossover + Mutate |   Uniform mask crossover (p=0.9)
|                     |   Bit-flip mutation (p=0.02 per feature)
|                     |   Blend crossover on hyperparams (alpha=0.3)
|                     |   Gaussian hyperparameter mutation (sigma=0.1)
+----------+----------+
           |
     Next Generation
           |
     (30 generations)
           |
           v
    Pareto Front of (feature subset, classifier) pairs
```

### Evolved Hyperparameters

NSGA-II jointly evolves feature masks and RandomForest hyperparameters:

| Hyperparameter | Search Range |
|---|---|
| n_estimators | 60 - 400 |
| max_depth | 2 - 24 |
| min_samples_split | 2 - 20 |
| min_samples_leaf | 1 - 10 |
| max_features | 0.15 - 1.0 (fraction) |

The framework also supports SVM (RBF/linear, C and gamma evolved in log space) and optionally XGBoost. The Pareto-optimal solution in this run converged on RandomForest.

---

## Experimental Setup

**Dataset:** Wisconsin Breast Cancer (scikit-learn built-in, UCI origin)

| Property | Value |
|---|---|
| Samples | 569 |
| Features | 30 (continuous) |
| Task | Binary classification: malignant vs. benign |
| Class balance | Balanced weighting (class_weight="balanced") |

**NSGA-II Configuration:**

| Parameter | Value |
|---|---|
| Population size | 64 |
| Generations | 30 |
| Crossover probability | 0.9 |
| Mutation probability (individual) | 0.3 |
| Bit-flip probability (per feature) | 0.02 |
| Hyperparameter mutation | Gaussian perturbation, sigma=0.1 |
| Crossover type | Uniform (feature mask) + Blend alpha=0.3 (hyperparameters) |
| Selection | NSGA-II via DEAP selNSGA2 |

---

## Results

### Pareto-Optimal Solution

The Pareto front converged to a solution that matches full-feature baselines with 70% of the feature space removed:

| Metric | Value |
|---|---|
| Test Accuracy | **94.74%** |
| Features Selected | **9 / 30** |
| Feature Reduction | **70%** |
| F1-Score | 0.958 |
| Precision | 0.971 |
| Recall | 0.944 |
| CV Training Accuracy | 96.05% |

**Evolved classifier configuration:**
- Algorithm: RandomForestClassifier
- n_estimators: 132
- max_depth: 24
- min_samples_split: 3
- min_samples_leaf: 4
- max_features: 0.57

### What the Pareto Front Shows

The front maps every discovered (feature count, accuracy) trade-off point from the minimal viable subset to the full feature set. A practitioner deploying to a resource-constrained environment can choose a point further left on the front. A practitioner where accuracy is critical can choose a point further right. Both choices are informed by the same optimization run.

---

## Baseline Comparison

| Method | Test Accuracy | Features | Notes |
|---|---|---|---|
| Grid Search RF | 94.74% | 30 | Exhaustive hyperparameter search, all features |
| RF Importance + RF | 94.74% | 30 | Importance-ranked, no feature reduction |
| RFE + RandomForest | 94.74% | 7 | Single solution, no trade-off curve |
| PCA + RandomForest | 93.86% | 10 components | Components, not original features - loses interpretability |
| Random Search RF | 91.23% | 30 | Suboptimal hyperparameters |
| **NSGA-II (this work)** | **94.74%** | **9** | **Full Pareto front, joint feature + hyperparameter optimization** |

NSGA-II matches the best baselines on accuracy while returning not a single solution but a complete trade-off curve - something no single-objective method can produce.

---

## Repository Structure

```
multi-objective-feature-selection/
|
+-- run_experiment.py   # Main NSGA-II implementation + experiment runner
+-- requirements.txt
+-- LICENSE
+-- README.md
```

---

## Installation

```bash
git clone https://github.com/royxforge/multi-objective-feature-selection.git
cd multi-objective-feature-selection

pip install -r requirements.txt
```

**Core dependencies:** scikit-learn · DEAP · NumPy · Matplotlib

---

## Usage

```bash
python run_experiment.py
```

The script loads the Wisconsin Breast Cancer dataset, runs NSGA-II for 30 generations across a population of 64, evaluates all Pareto-optimal solutions on the held-out test set, and plots the final Pareto front in objective space (accuracy vs. feature count).

To swap the dataset, replace the data loading block with any binary classification dataset in sklearn-compatible format (X, y arrays). The optimizer is dataset-agnostic.

---

## Research Context

Multi-objective feature selection is an active research area in medical ML, where interpretability requirements structurally conflict with accuracy maximization. Regulatory frameworks (EU AI Act, FDA guidance on clinical decision support) increasingly require explainability, which directly rewards feature economy.

NSGA-II (Deb et al., 2002) is one of the most cited multi-objective evolutionary algorithms, with strong theoretical guarantees on Pareto front convergence and diversity preservation via crowding distance. This implementation extends the standard NSGA-II formulation by jointly evolving feature masks and classifier hyperparameters in a single optimization loop - avoiding the two-stage bias introduced by selecting features first and tuning hyperparameters second.

**Connection to related work:** The unsupervised confidence metric in [Unsupervised Confidence Estimation
](https://github.com/royxforge/unsupervised-confidence-estimation) and the drift monitoring in [Production Drift Detection
](https://github.com/royxforge/production-drift-detection) both operate on model outputs. Feature selection determines what goes into the model. These three projects form a coherent pipeline: select features carefully, monitor for drift, and quantify uncertainty in deployment. [Unified Time Series Foundation Model](https://github.com/royxforge/unified-time-series-foundation-model) extends the same philosophy to time series foundation models, where the 15 meta-features used for model selection are a forecasting-specific analog of the feature economy question studied here.

---

## Citation

```bibtex
@software{roy2025nsga2featureselection,
  author = {Roy, Sourav},
  title  = {multi-objective-feature-selection: NSGA-II for Medical Tabular Data},
  year   = {2026},
  url    = {https://github.com/royxforge/multi-objective-feature-selection}
}
```

---

<p align="center">
  <sub>Built by <a href="https://github.com/royxforge">Sourav Roy</a> · Artificial Intelligence Engineer · Accure Inc.</sub>
</p>
