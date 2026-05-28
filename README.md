# 🚀 30 AI & Machine Learning Algorithms in Python

A beautifully curated, zero-boilerplate repository containing **30 core Machine Learning, Deep Learning, and Reinforcement Learning algorithms** implemented in standalone Python scripts. 

Each script is completely self-contained, loads a standard/synthetic dataset, trains the model, outputs performance metrics, and saves a high-quality visualization to the `outputs/` directory.

---

<p align="center">
  <img src="_img/30-AI-Algos.png" alt="30 AI & Machine Learning Algorithms Map" width="100%" />
</p>

---

## 🌟 Key Features

* **Zero-Boilerplate Standalone Scripts:** Every file is self-contained. No complex internal modules to import—just run the script!
* **Automatic Visualizations:** Every algorithm generates and saves highly informative plots (e.g., decision boundaries, training curves, loss landscapes) in the `outputs/` folder.
* **Modern Tech Stack:** Utilizes `scikit-learn`, `PyTorch`, `XGBoost`, `matplotlib`, `seaborn`, and `numpy`.
* **Fast Setup:** Full support for standard `pip` virtual environments as well as ultra-fast `uv` workspace synchronization.

---

## 📂 Repository Directory Structure

```text
30-ML-Algos/
├── _img/
│   └── 30-AI-Algos.png          # Overview roadmap image
├── outputs/                     # Target directory for generated plots & visualizations
├── 01_linear_regression.py      # Self-contained executable algorithm scripts
├── 02_logistic_regression.py
│   ...
├── 30_genetic_algorithms.py
├── requirements.txt             # List of dependencies for pip
├── pyproject.toml               # Modern Python project configuration
├── uv.lock                      # uv lockfile for exact environment reproducibility
└── README.md                    # Project documentation
```

---

## 🛠️ Installation & Setup

Set up the project environment using `uv` (a fast alternative to `pip` and `venv`):

```bash
# (⭐ Only run if starting a NEW project, not when cloning an existing one)
uv init                        # Initialize a new project with pyproject.toml

# Create a virtual environment (like python -m venv but faster)
uv venv                        # Makes a .venv folder in your project

# Activate the virtual environment
# 👉 Use the one matching your OS:
.venv\Scripts\activate         # Windows (PowerShell / CMD)
source .venv/bin/activate      # macOS / Linux

# Install dependencies from requirements.txt
# (use when project already has a requirements.txt file)
uv add -r requirements.txt     

# Install ipykernel for Jupyter Notebook / VSCode integration
# (use if you need to run the project in Jupyter or interactive notebooks)
uv add ipykernel
```

*Note: PyTorch is used for deep learning scripts. If XGBoost is unavailable on your system, the `04_xgboost.py` script automatically falls back to scikit-learn's standard `GradientBoostingClassifier`.*


---

## 🗺️ Index of Algorithms

| # | Category | Algorithm | Script | Key Libraries | Description |
|---|---|---|---|---|---|
| **01** | **Regression** | Linear Regression | [`01_linear_regression.py`](01_linear_regression.py) | `scikit-learn` | Fits continuous values based on features (Diabetes dataset). |
| **02** | **Regression** | Logistic Regression | [`02_logistic_regression.py`](02_logistic_regression.py) | `scikit-learn` | Models binary classification boundaries using sigmoid activations. |
| **03** | **Trees & Ensembles** | Decision Tree | [`03_decision_tree.py`](03_decision_tree.py) | `scikit-learn` | Builds tree-structured classifiers recursively partitioning features. |
| **04** | **Trees & Ensembles** | XGBoost | [`04_xgboost.py`](04_xgboost.py) | `xgboost` / `scikit-learn` | High-efficiency gradient boosting classifier with feature importances. |
| **05** | **Supervised Learning** | Support Vector Machine | [`05_support_vector_machine.py`](05_support_vector_machine.py) | `scikit-learn` | Maximizes margin boundaries using standard SVM kernel methods. |
| **06** | **Trees & Ensembles** | Random Forest | [`06_random_forest.py`](06_random_forest.py) | `scikit-learn` | Bootstrap aggregation of decision trees to prevent overfitting. |
| **07** | **Trees & Ensembles** | AdaBoost | [`07_adaboost.py`](07_adaboost.py) | `scikit-learn` | Fits adaptive weak learners sequentially weighting misclassifications. |
| **08** | **Trees & Ensembles** | Gradient Boosting | [`08_gradient_boosting.py`](08_gradient_boosting.py) | `scikit-learn` | Fits trees on residual gradients of loss functions. |
| **09** | **Supervised Learning** | Naive Bayes | [`09_naive_bayes.py`](09_naive_bayes.py) | `scikit-learn` | Probabilistic classification based on independent Bayes theorem. |
| **10** | **Clustering** | Hierarchical Clustering | [`10_hierarchical_clustering.py`](10_hierarchical_clustering.py) | `scipy` / `scikit-learn` | Generates agglomerative cluster trees and draws dendrograms. |
| **11** | **Clustering** | K-Means | [`11_k_means_clustering.py`](11_k_means_clustering.py) | `scikit-learn` | Partitions observations into $K$ spherical clusters iteratively. |
| **12** | **Supervised Learning** | K-Nearest Neighbors | [`12_k_nearest_neighbors.py`](12_k_nearest_neighbors.py) | `scikit-learn` | Classifies based on majority votes of closest sample points. |
| **13** | **Dimension Reduction** | PCA | [`13_principal_component_analysis.py`](13_principal_component_analysis.py) | `scikit-learn` | Orthogonal projection reducing dimensions to maximize variance. |
| **14** | **Clustering** | DBSCAN | [`14_dbscan.py`](14_dbscan.py) | `scikit-learn` | Density-based grouping highlighting cluster structures of arbitrary shapes. |
| **15** | **Dimension Reduction** | t-SNE | [`15_t_sne.py`](15_t_sne.py) | `scikit-learn` | Manifold projection mapping high-dim distributions to 2D space. |
| **16** | **Reinforcement Learning**| Tabular Q-Learning | [`16_q_learning.py`](16_q_learning.py) | `numpy` | Model-free off-policy value iteration in gridworld environments. |
| **17** | **Reinforcement Learning**| SARSA | [`17_sarsa.py`](17_sarsa.py) | `numpy` | Model-free on-policy control updates following selected actions. |
| **18** | **Reinforcement Learning**| Deep Q-Network (DQN) | [`18_deep_q_network.py`](18_deep_q_network.py) | `pytorch` | Uses deep networks to approximate value estimates with experience replay. |
| **19** | **Reinforcement Learning**| Policy Gradient | [`19_policy_gradient.py`](19_policy_gradient.py) | `pytorch` | Directly optimizes actions via probability weights (REINFORCE). |
| **20** | **Reinforcement Learning**| Actor-Critic | [`20_actor_critic.py`](20_actor_critic.py) | `pytorch` | Combines value function critics with direct policy actor updates. |
| **21** | **Deep Learning** | ANN | [`21_artificial_neural_network.py`](21_artificial_neural_network.py) | `pytorch` | Fully-connected multi-layer feedforward neural network classifier. |
| **22** | **Deep Learning** | CNN | [`22_convolutional_neural_network.py`](22_convolutional_neural_network.py) | `pytorch` | Spatial grid feature extraction using 2D convolutional filters. |
| **23** | **Deep Learning** | RNN | [`23_recurrent_neural_network.py`](23_recurrent_neural_network.py) | `pytorch` | Processes sequential time series with recurrent state feedback. |
| **24** | **Deep Learning** | LSTM | [`24_long_short_term_memory.py`](24_long_short_term_memory.py) | `pytorch` | Employs input, output, and forget gates to solve vanishing gradients. |
| **25** | **Deep Learning** | Transformer | [`25_transformer.py`](25_transformer.py) | `pytorch` | Uses self-attention layers to model sequences and dependencies. |
| **26** | **Clustering** | K-Means++ | [`26_k_means_plus_plus.py`](26_k_means_plus_plus.py) | `scikit-learn` | Smarter initialization choosing distant starting centroids. |
| **27** | **Deep Learning** | Autoencoder | [`27_autoencoders.py`](27_autoencoders.py) | `pytorch` | Unsupervised dimensionality reduction by learning encoding/decoding loops. |
| **28** | **Anomaly Detection** | Isolation Forest | [`28_isolation_forest.py`](28_isolation_forest.py) | `scikit-learn` | Isolates anomalous observations using recursive random partitions. |
| **29** | **Reinforcement Learning**| MDP | [`29_markov_decision_process.py`](29_markov_decision_process.py) | `numpy` | Solves grid environments using value iteration equations. |
| **30** | **Evolutionary Algos** | Genetic Algorithm | [`30_genetic_algorithms.py`](30_genetic_algorithms.py) | `numpy` | Population optimization using selection, crossover, and mutation. |


---

## Contributing and Support

Contributions are welcome. Please open an issue before submitting major changes.

If this repository helps you, consider giving it a star so other learners can discover it.

<div align="center">
  <br/>
  <p><b>Connect with me</b></p>
  <a href="https://twitter.com/F4izy">
    <img src="https://img.shields.io/badge/Twitter-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white" alt="Twitter"/>
  </a>
  <a href="https://www.linkedin.com/in/mohd-faizy/">
    <img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn"/>
  </a>
  <a href="https://github.com/mohd-faizy">
    <img src="https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white" alt="GitHub"/>
  </a>
</div>
