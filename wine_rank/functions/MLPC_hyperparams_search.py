from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
import optuna
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

def MLP_hyperparams_search(X_train, y_train, X_test, y_test, n_trials=100, type_of_search='random'):
    
    # Hyperparameter grid for GridSearchCV and RandomizedSearchCV
    param_grid = {
        'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50), (100, 100)],
        'activation': ['relu', 'tanh', 'logistic'],
        'solver': ['adam', 'sgd', 'lbfgs'],
        'learning_rate_init': np.logspace(-4, -1, 10),
        'alpha': np.logspace(-5, -1, 10),
        'max_iter': [200, 300, 500],
        'batch_size' : [32, 64, 128],
        'momentum' : np.linspace(0.8,0.99,10)
    }
    
    # Optuna optimization function
    def objective(trial):
        """Optimization function for Optuna."""
        
        params = {
            'hidden_layer_sizes': trial.suggest_categorical('hidden_layer_sizes', [(50,), (100,), (50, 50), (100, 50), (100, 100)]),
            'activation': trial.suggest_categorical('activation', ['relu', 'tanh', 'logistic']),
            'solver': trial.suggest_categorical('solver', ['adam', 'sgd', 'lbfgs']),
            'learning_rate_init': trial.suggest_float('learning_rate_init', 0.0001, 0.1, log=True),
            'alpha': trial.suggest_float('alpha', 0.00001, 0.1, log=True),
            'max_iter': trial.suggest_int('max_iter', 200, 500),
            'batch_size' : trial.suggest_categorical('batch_size' , [32, 64, 128]),
          }
        if params['solver'] == 'sgd':
          params['momentum'] = trial.suggest_float('momentum',0.8,0.99)
            
        model = MLPClassifier(**params, random_state=42)
        
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1)
        return cv_scores.mean() # Return mean cross validation score


    if type_of_search == 'grid':
      search = GridSearchCV(
            MLPClassifier(random_state=42),
            param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1
        )
    elif type_of_search == 'random':
        search = RandomizedSearchCV(
            MLPClassifier(random_state=42),
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
        model = MLPClassifier(**best_params, random_state=42)
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