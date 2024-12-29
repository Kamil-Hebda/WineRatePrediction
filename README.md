# Wine Rate Prediction

---

## Overview
The **Wine Rate Prediction** project leverages machine learning to predict the quality ratings of wine based on its physicochemical attributes. This repository contains the code and necessary instructions to preprocess the dataset, tune machine learning models, and evaluate their performance for this task.

The primary goal is to explore various machine learning approaches, optimize model performance, and provide reproducible results for wine quality prediction.

---

## Features
- Implements hyperparameter optimization using **GridSearchCV**, **RandomizedSearchCV**, and **Optuna**.
- Supports Logistic Regression, XGBoost Classifier and MLP Classifier
- Provides a Jupyter Notebook interface for experimentation.
- Uses a comprehensive dataset for white and red wine, licensed under **Creative Commons Attribution 4.0 International (CC BY 4.0)**.

---

## Dataset

### Description
The dataset contains physicochemical attributes of red and white wine samples. The data was sourced from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Wine+Quality) and is credited to the following authors:

**Creators:**
- Paulo Cortez
- A. Cerdeira
- F. Almeida
- T. Matos
- J. Reis

**DOI**: [10.24432/C56S3T](https://doi.org/10.24432/C56S3T)

**License**:  
This dataset is licensed under the **Creative Commons Attribution 4.0 International (CC BY 4.0)**. You can share and adapt the data for any purpose, provided appropriate credit is given.

---

### Attributes and Data Types

| Attribute               | Description                             | Data Type  |
|--------------------------|-----------------------------------------|------------|
| **Fixed acidity**        | Tartaric acid concentration            | `float`    |
| **Volatile acidity**     | Acetic acid concentration              | `float`    |
| **Citric acid**          | Citric acid concentration              | `float`    |
| **Residual sugar**       | Sugar left after fermentation          | `float`    |
| **Chlorides**            | Salt content                           | `float`    |
| **Free sulfur dioxide**  | Free form of SO₂ in wine               | `int`      |
| **Total sulfur dioxide** | Total amount of SO₂ in wine            | `int`      |
| **Density**              | Density of the wine                    | `float`    |
| **pH**                   | pH value of the wine                   | `float`    |
| **Sulphates**            | Potassium sulphate concentration       | `float`    |
| **Alcohol**              | Alcohol percentage                     | `float`    |
| **Quality**              | Wine quality rating (target variable)  | `int`      |

---

## Instructions for Downloading the Dataset

1. Go to the [Wine Quality Dataset on UCI Repository](https://archive.ics.uci.edu/ml/datasets/Wine+Quality).
2. Download the `winequality-red.csv` and `winequality-white.csv` files.
3. Place the files in the `data/raw/` directory of this repository.

---

## Setup and Installation

### Prerequisites
- Python 3.8 or higher
- pip
- git

### Clone the Repository
```bash
git clone https://github.com/Kamil-Hebda/WineRatePrediction.git
cd WineRatePrediction
```

### Install Dependencies
Run the following command to install the required Python packages:
```bash
pip install -r requirements.txt
```

---

## How to Use

### Preprocessing the Data
The script preprocesses the dataset by:
- Combining red and white wine data.
- Using `ColumnTransformer` to preprocessing data using `MinMaxScaler` and `BinaryEncoder`
- Splitting the data into training and testing sets.

### Hyperparameter Optimization
The `LR_hyperparams_search` function supports three types of hyperparameter searches:
- **Grid Search** (`grid`)
- **Random Search** (`random`)
- **Optuna** (`optuna`)

### Run the Optimization
You can run the optimization using the widget interface in Jupyter Notebook or by calling the function directly:
```python
best_model = LR_hyperparams_search(X_train, y_train, X_test, y_test, n_trials=50, type_of_search='optuna')
```

### Evaluate Model Performance
Evaluate the trained model using accuracy and other metrics:
```python
TO DO
```

---

## Widgets Interface
An interactive widget interface is available for selecting the model type, search type, and number of trials for hyperparameter tuning. Launch it by running the `03_model_training.ipynb` notebook.

---

## Project Structure
```
WineRatePrediction/
│
├── data/
|   |──raw/                   # Folder to store downloaded datasets
|   |   |──wine+quality/
│   |   |   ├── winequality-red.csv
│   |   |   ├── winequality-white.csv
|   |──processed/                   # Folder to store processed data
│
├── notebooks/                # Jupyter Notebooks for experimentation
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│
├── functions/                  # Python scripts
│   ├──__init__.py             # Functions for hyperparameter optimization
│   ├──LR_hyperparams_search.py
│   ├──MLPC_hyperparams_search.py
│   ├──xgboost_hyperparams_search.py
│
├── requirements.txt          # Required Python packages
├── README.md                 # Project documentation
```

---

## Results
The best-performing model and its parameters are displayed in the console after the optimization process. Detailed results can be logged for further analysis.

---

## Acknowledgments
This project was inspired by the dataset created by Paulo Cortez, A. Cerdeira, F. Almeida, T. Matos, and J. Reis, and hosted on the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Wine+Quality).

---

## License
This project is licensed under the **MIT License**. The dataset is licensed under the **Creative Commons Attribution 4.0 International (CC BY 4.0)**.
