# 2025-ITU-MachineLearning-Project-Claims-Risk-zrt
This repository contains the Machine Learning exam project for the BSc Data Science program (Fall 2025).  
The project investigates claims risk modeling in automobile insurance using various machine learning methods.
## Documentation

- [Project Guidelines](docs/ML_Project_Proposal_2025.pdf) – original project description from ITU.


## Project Workflow
1. **Data Cleaning/Exploratory Data Analysis (EDA)**  – Data cleaning, handling missing values.– Visualize distributions, detect outliers, study correlations, 
2. **PCA and clustering**
3. **Preprocessing/Feature Engineering** –Eencoding categorical features, scaling numeric variables. Create, transform, or select features to improve model performance.  
4. **Model Implementation (from scratch)** – Implement Decision Tree and Feed-Forward Neural Network manually using NumPy/SciPy.  
5. **Reference Models** – Use scikit-learn or other libraries to verify correctness and compare performance.  
6. **Model 3 (custom)** – Implement an additional method of your choice (e.g., ensemble, regression, or another ML algorithm).  
7. **Evaluation & Comparison** – Compare models using MAE, RMSE, and Poisson deviance; visualize results.  
8. **Report Writing** – Summarize methods, results, interpretations, and conclusions in the final report.


## Project Structure

The repository could be organized as:

2025-ITU-MachineLearning-Project-Claims-Risk-zrt/
│
├── data/
│   ├── raw/
│   │   ├── claims_train.csv
│   │   ├── claims_test.csv
│   │   └── README.md           # Describe dataset source, size, and columns
│   ├── processed/
│   │   └── cleaned_data.csv
│   └── external/               # (if any additional data sources used)
│
├── notebooks/
│   ├── 01_Preprocessing_EDA.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_models_scratch.ipynb # Implementations of M1 & M2 (from scratch)
│   ├── 04_models_reference.ipynb # Using libraries (e.g., sklearn, keras)
│   ├── 05_model3_experiment.ipynb # M3 (custom / advanced method)
│   └── 06_results_analysis.ipynb  # Comparison and metrics
│
├── src/
│   ├── data/
│   │   ├── load_data.py
│   │   ├── preprocess.py
│   │   └── utils.py
│   ├── models/
│   │   ├── decision_tree_scratch.py
│   │   ├── neural_network_scratch.py
│   │   ├── model3.py
│   │   └── evaluation.py
│   └── visualization/
│       ├── pca_plot.py
│       ├── clustering_plot.py
│       └── eda_charts.py
│
├── reports/
│   ├── ML_Project_Report.pdf   # final 10-page submission
│   └── figures/
│       ├── pca_plot.png
│       ├── clustering_map.png
│       └── model_results.png
│
├── docs/
│   └──ML_Project_Proposal_2025.pdf # project guidelines
│
├── tests/
│   ├── test_decision_tree.py
│   ├── test_neural_net.py
│   └── test_utils.py
│
├── requirements.txt
├── environment.yml             # optional: for conda reproducibility
├── .gitignore
├── README.md                   # summary, structure, how to run
└── LICENSE                     # optional if you open source it

