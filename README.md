# 2025-ITU-MachineLearning-Project-Claims-Risk-zrt
This repository contains the Machine Learning exam project for the BSc Data Science program (Fall 2025).  
The project investigates claims risk modeling in automobile insurance using various machine learning methods.
## Documentation

- [Project Guidelines](docs/ML_Project_Proposal_2025.pdf) – original project description from ITU.


## Project Structure

The repository could be organized as:

~~~
ML_Insurance_Project/
│
├── data/
│   ├── raw/                  # Original CSV files (immutable)
│   │   ├── claims_train.csv
│   │   └── claims_test.csv
│   └── processed/            # Cleaned and preprocessed datasets
│
├── src/                      # Core Python modules (reusable)
│   ├── __init__.py
│   ├── preprocessing.py      # Data cleaning and feature engineering
│   ├── metrics.py            # Evaluation metrics (RMSE, MAE, etc.)
│   ├── utils.py              # Helper functions (plotting, saving results)
│   └── models/
│       ├── __init__.py
│       ├── decision_tree.py  # Decision Tree from scratch + library wrapper
│       ├── neural_network.py # Feed-Forward Neural Network from scratch + library wrapper
│       └── other_models.py   # Additional models (Random Forest, XGBoost, etc.)
│
├── experiments/              # Scripts to reproduce experiments
│   ├── run_eda.py            # Exploratory Data Analysis and visualizations
│   ├── run_pca_clustering.py # PCA & clustering analysis
│   ├── run_models.py         # Train & evaluate all models
│   └── hyperparameter_search.py # Optional hyperparameter tuning scripts
│
├── results/                  # Output of experiments
│   ├── figures/              # Plots and visualizations
│   └── metrics/              # CSV/JSON files with evaluation scores
│
├── report/
│   └── ML_Project_Report.pdf # Final report (≤10 pages)
│
├── requirements.txt          # Python dependencies
├── README.md                 # Project overview and instructions
└── .gitignore                # Ignore unnecessary files (e.g., __pycache__, .ipynb_checkpoints)
~~~

---