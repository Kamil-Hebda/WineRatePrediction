import xgboost as xgb
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import optuna
import numpy as np
from sklearn.metrics import accuracy_score
import time 

def xgboost_hyperparams_search(X_train, y_train, X_test, y_test, n_trials=100, type_of_search='random'):
    
    # Definition of hyperparameters grid
    param_grid = {
        'objective': ['multi:softmax'],
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 10],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.7, 0.8, 1.0],
        'colsample_bytree': [0.7, 0.8, 1.0]
    }
    
    # Function for Optuna optimization
    def objective(trial):
        param = {
            'objective': 'multi:softmax',
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
            'subsample': trial.suggest_float('subsample', 0.7, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
        }
        model = xgb.XGBClassifier(**param)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy
    
    if type_of_search == 'grid':
        search = GridSearchCV(xgb.XGBClassifier(), param_grid, cv=5, scoring='accuracy')
    elif type_of_search == 'random':
        search = RandomizedSearchCV(xgb.XGBClassifier(), param_grid, n_iter=n_trials, cv=5, scoring='accuracy', random_state=42)
    elif type_of_search == 'optuna':
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        best_params = study.best_params
        model = xgb.XGBClassifier(**best_params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Best Parameters: {best_params}")
        print(f"Accuracy: {accuracy:.4f}")
        return model
    else:
        raise ValueError("type_of_search must be 'grid', 'random', or 'optuna'")
    
    time_start = time.time()
    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    
    y_pred = best_model.predict(X_test)

    time_end = time.time()
    
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Best Parameters: {search.best_params_}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Time: {time_end - time_start:.2f} seconds")
    
    return best_model