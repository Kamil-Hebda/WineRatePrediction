from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
import optuna
import numpy as np
from sklearn.metrics import accuracy_score
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Nadam
import time
from tensorflow.keras.callbacks import EarlyStopping
from scikeras.wrappers import KerasClassifier

def create_mlp_functional(input_dim, output_dim, hidden_units1=128, hidden_units2=None, activation='relu', learning_rate=0.001, optimizer='adam', dropout=0.1):
    inputs = Input(shape=(input_dim,))
    x = Dense(hidden_units1, activation=activation)(inputs)
    if dropout > 0.0 or hidden_units2 is not None:
        x = Dropout(dropout)(x)
    if hidden_units2 is not None:
        x = Dense(hidden_units2, activation=activation)(x)
    outputs = Dense(output_dim, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    if optimizer == 'Adam':
        optimizer = Adam(learning_rate=learning_rate)
    elif optimizer == 'SGD':
        optimizer = SGD(learning_rate=learning_rate)
    elif optimizer == 'RMSprop':
        optimizer = RMSprop(learning_rate=learning_rate)
    elif optimizer == 'Nadam':
        optimizer = Nadam(learning_rate=learning_rate)

    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

def MLP_model_hyperparams_search(X_train, y_train, X_test, y_test, n_trials=10, type_of_search='random'):

    # Definition of early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Definition of hyperparameters grid
    param_grid = {
        'hidden_units1': [64, 128, 256],
        'hidden_units2': [None, 32, 64, 128],  # None means no second layer
        'activation': ['relu', 'tanh'],
        'learning_rate': [0.001, 0.01, 0.1],
        'optimizer': ['Adam', 'SGD', 'RMSprop', 'Nadam'],
        'dropout': [0.1, 0.2, 0.5]  # Dropout with rate 0 is no dropout
    }

    # Function for Optuna optimization
    def objective(trial):
        param = {
            'hidden_units1': trial.suggest_categorical('hidden_units1', [64, 128, 256]),
            'hidden_units2': trial.suggest_categorical('hidden_units2', [None, 32, 64, 128]),
            'activation': trial.suggest_categorical('activation', ['relu', 'tanh']),
            'learning_rate': trial.suggest_float('learning_rate', 0.0001, 0.1, log=True),
            'optimizer': trial.suggest_categorical('optimizer', ['Adam', 'SGD', 'RMSprop', 'Nadam']),
            'dropout': trial.suggest_categorical('dropout', [0.1, 0.2, 0.5]),
        }

        model = create_mlp_functional(
            input_dim=X_train.shape[1],
            output_dim=len(np.unique(y_train)),
            **param
        )

        # Splitting the data into training and validation sets
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        model.fit(X_train_split, y_train_split, epochs=10, verbose=0, callbacks=[early_stopping], validation_data=(X_val_split, y_val_split))
        y_pred = model.predict(X_test)
        y_pred = np.argmax(y_pred, axis=1)
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy
    
    def build_model(hidden_units1, hidden_units2, activation, learning_rate, optimizer, input_dim, output_dim, dropout, early_stopping):
        return KerasClassifier(model=create_mlp_functional,
                       input_dim=input_dim,
                       output_dim=output_dim,
                       hidden_units1=hidden_units1,
                       hidden_units2=hidden_units2,
                       activation=activation,
                       optimizer=optimizer,
                       learning_rate=learning_rate,
                       dropout=dropout,
                       epochs=10,
                       batch_size=32,
                       verbose=0,
                       callbacks=[early_stopping],
                       validation_split=0.2,
                       )

    if type_of_search == 'grid':
         search = GridSearchCV(
            estimator=build_model(128, None, 'relu', 0.001, 'Adam', X_train.shape[1], len(np.unique(y_train)),dropout=0.1, early_stopping= early_stopping),
            param_grid=param_grid,
            cv=None,
            scoring='accuracy',
        )
    elif type_of_search == 'random':
        search = RandomizedSearchCV(
           estimator=build_model(128, None, 'relu', 0.001, 'Adam', X_train.shape[1], len(np.unique(y_train)), dropout=0.1, early_stopping= early_stopping),
            param_distributions=param_grid,
            n_iter=n_trials,
            cv=None,
            scoring='accuracy',
            random_state=42
        )
    elif type_of_search == 'optuna':
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        best_params = study.best_params
        model = create_mlp_functional(
            input_dim=X_train.shape[1],
            output_dim=len(np.unique(y_train)),
            **best_params
        )
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        model.fit(X_train_split, y_train_split, epochs=10, verbose=0, callbacks=[early_stopping], validation_data=(X_val_split, y_val_split))
        y_pred = model.predict(X_test)
        y_pred = np.argmax(y_pred, axis=1)
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
    #y_pred = np.argmax(y_pred, axis=1)

    time_end = time.time()

    accuracy = accuracy_score(y_test, y_pred)

    print(f"Best Parameters: {search.best_params_}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Time: {time_end - time_start:.2f} seconds")

    return best_model