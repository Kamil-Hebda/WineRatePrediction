from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
import optuna
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

def LR_hyperparams_search(X_train, y_train, X_test, y_test, n_trials=100, type_of_search='random'):
    
    # Hyperparameter grid for GridSearchCV and RandomizedSearchCV
    param_grid = {
        'penalty': ['l1', 'l2', 'elasticnet'],
        'C': np.logspace(-4, 4, 20),
        'solver': ['liblinear', 'saga'],  # Restrict to compatible solvers
        'max_iter': [1000, 2500, 5000],
        'l1_ratio': np.linspace(0, 1, 10)  # Used only with elasticnet
    }

    # Adjust solver for 'elasticnet' penalty to be 'saga' only
    if 'elasticnet' in param_grid['penalty']:
        param_grid['solver'] = ['saga']  # Only saga solver can handle elasticnet

    # Optuna optimization function
    def objective(trial):
        """Optimization function for Optuna."""
        penalty = trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet'])
        solver = trial.suggest_categorical('solver', ['liblinear', 'saga'])  # Compatible solvers only

        # Adjust incompatible combinations
        if penalty == 'elasticnet' and solver != 'saga':
            solver = 'saga'  # Force solver to 'saga' if 'elasticnet' is chosen

        params = {
            'penalty': penalty,
            'C': trial.suggest_float('C', 0.0001, 100.0, log=True),
            'solver': solver,
            'max_iter': trial.suggest_int('max_iter', 1000, 5000),
        }
        if penalty == 'elasticnet':
            params['l1_ratio'] = trial.suggest_float('l1_ratio', 0.0, 1.0)
        
        model = LogisticRegression(**params)
        
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        return cv_scores.mean() # Return mean cross validation score


    if type_of_search == 'grid':
        # Filter grid for compatible combinations
        param_grid_filtered = {
            'penalty': ['l1', 'l2', 'elasticnet'],
            'C': np.logspace(-4, 4, 20),
            'solver': ['liblinear', 'saga'],  # Compatible solvers
            'max_iter': [1000, 2500, 5000],
            'l1_ratio': np.linspace(0, 1, 10)  # Used only with elasticnet
        }
        param_grid_filtered['solver'] = ['saga']  # Only saga solver can handle elasticnet

        search = GridSearchCV(
            LogisticRegression(),
            param_grid_filtered,
            cv=5,
            scoring='accuracy',
            n_jobs=-1
        )
    elif type_of_search == 'random':
        search = RandomizedSearchCV(
            LogisticRegression(),
            param_distributions=param_grid,
            n_iter=n_trials,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            random_state=42
        )
    elif type_of_search == 'optuna':
        # Use Optuna for hyperparameter optimization
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        best_params = study.best_params
          # Ensure compatibility for Optuna
        if best_params['penalty'] == 'elasticnet' and best_params['solver'] != 'saga':
            best_params['solver'] = 'saga'
        model = LogisticRegression(**best_params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Best Parameters: {best_params}")
        print(f"Accuracy: {accuracy:.4f}")
        return model
    else:
        raise ValueError("type_of_search must be 'grid', 'random', or 'optuna'")

     # Fit for GridSearchCV or RandomizedSearchCV
    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Best Parameters: {search.best_params_}")
    print(f"Accuracy: {accuracy:.4f}")

    # Train best model
    best_model.fit(X_train, y_train)
    return best_model