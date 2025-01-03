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

### Data Loading and Exploration

1. **Load the dataset**: Place the red and white wine datasets in the `data/raw/` directory and load them using:
   ```python
   red_wine_df = pd.read_csv("data/raw/wine+quality/winequality-red.csv", sep=';')
   white_wine_df = pd.read_csv("data/raw/wine+quality/winequality-white.csv", sep=';')

   red_wine_df['colour'] = 'red'
   white_wine_df['colour'] = 'white'

   dataset_df = pd.concat([red_wine_df, white_wine_df])
   ```

2. **Explore the data**:
   - Check basic information: `dataset_df.info()`
   - View statistics: `dataset_df.describe()`
   - Inspect unique values in categorical columns: `dataset_df.select_dtypes(include=['object']).nunique()`

3. **Visualize data** (optional):
   - Plot histograms: `dataset_df.hist(figsize=(10, 10))`
   - Examine correlations:
     ```python
     sns.heatmap(dataset_df.corr(), annot=True, cmap='coolwarm')
     ```

---

### Preprocessing the Data

1. **Prepare the data**:
   - Shuffle: `dataset_df = dataset_df.sample(frac=1).reset_index(drop=True)`
   - Split target and features:
     ```python
     y = dataset_df['quality']
     X = dataset_df.drop(['quality'], axis=1)
     ```

2. **Define column types**:
   ```python
   numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
   categorical_features = X.select_dtypes(include=['object']).columns.tolist()
   ```

3. **Split into training and test sets**:
   ```python
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
   ```

4. **Apply preprocessing**:
   - Scale numerical features and encode categorical ones:
     ```python
     preprocessor = ColumnTransformer([
         ('num', MinMaxScaler(), numerical_features),
         ('cat', ce.BinaryEncoder(), categorical_features)
     ])
     X_train_processed = preprocessor.fit_transform(X_train)
     X_test_processed = preprocessor.transform(X_test)
     ```

5. **Save preprocessed data**:
   ```python
   pd.DataFrame(X_train_processed).to_csv("data/processed/processed_data_train.csv", index=False)
   pd.DataFrame(X_test_processed).to_csv("data/processed/processed_data_test.csv", index=False)
   ```

---

### Training and Hyperparameter Optimization

1. **Hyperparameter Optimization**:
   Search for the best parameters using methods like grid search or Optuna:
   ```python
   best_model = LR_hyperparams_search(X_train, y_train, X_test, y_test, n_trials=50, type_of_search='random')
   ```

2. **Train models**:
   Train models (e.g., Logistic Regression, XGBoost, MLP)

3. **Interactive Optimization**:
   Use a widget interface to select models and tune hyperparameters directly in Jupyter Notebook.
    You can run the optimization using the widget interface in Jupyter Notebook: 
    ![image](https://github.com/user-attachments/assets/a685fa1c-8f6c-4df8-b069-3111ad81944d)
---



### Evaluate Model Performance

After training and optimizing the models, you can evaluate their performance using various metrics. These metrics help assess how well the model predicts wine quality, both in terms of classification and regression.

For **classification** tasks, we use precision, recall, F1 score, and accuracy:
- **Precision** measures the proportion of positive predictions that were correct.
- **Recall** measures the proportion of actual positives that were correctly identified by the model.
- **F1 Score** is the harmonic mean of precision and recall, balancing both metrics.
- **Accuracy** is the proportion of correct predictions (both positive and negative) out of all predictions.

For **regression** tasks, we evaluate using Mean Squared Error (MSE), Mean Absolute Error (MAE), and R-squared (R²):
- **MSE** measures the average squared difference between predicted and actual values, giving more weight to large errors. 
- **MAE** measures the average absolute difference between predicted and actual values, indicating the magnitude of error without amplifying large deviations.
- **R²** indicates how well the model's predictions match the actual data; values closer to 1 suggest a better fit.

These metrics are especially useful for regression problems, but I also use them here to observe how far the model's predictions deviate from the actual wine quality classes. For example, an MSE of 0.4 indicates minor errors, suggesting that the predictions are close to the true values. However, an MSE of 3 would reflect significant errors, where the predicted quality is far from the actual value. I believe these metrics provide a clearer understanding of the model's predictions and help evaluate its overall performance more effectively.

You can evaluate the performance as follows:
```python
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Accuracy: {accuracy:.4f}")

print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R-squared (R2): {r2:.4f}
")
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
|   |──processed/             # Folder to store processed data
│
├── notebooks/                # Jupyter Notebooks for experimentation
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│
├── functions/                # Python scripts
│   ├──__init__.py            # Initialize Python module
│   ├──LR_hyperparams_search.py
│   ├──MLPC_hyperparams_search.py
│   ├──xgboost_hyperparams_search.py
│   ├──MLP_model.py           # Custom MLP model implementation
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
