from utils import split_data
from sklearn.metrics import accuracy_score, mean_absolute_percentage_error, mean_absolute_error
import xgboost as xgb
import pandas as pd
from utils import OptunaObjective
from sklearn.impute import SimpleImputer
import numpy as np


# User defined variables (hyperparameters space can be modified according to needs)
DATA_NAME = 'data_duplicate_removed'
RANDOM_STATE = 42
MODEL = xgb.XGBRegressor 
N_TRIALS = 50
VALIDATION_STRATEGY = 'validation'
MODEL_NAME = "MSE_ln_50"
MODEL_SAVE_NAME = f"./src/1st_task/models/{MODEL_NAME}.model"

# Load the dataset
df = pd.read_csv(f'./description/{DATA_NAME}.csv')


# Imputing the missing values with zero, instead of erasing the columns with >98% missing values produce better results
# better results are obtained even in the case that the missing values are passed to the network (trees can handle missing values)
imputer = SimpleImputer(strategy='constant', fill_value=0)  # Initialize the SimpleImputer with strategy='constant' and fill_value=0
imputed_data = imputer.fit_transform(df)  # Fit the imputer to the DataFrame and transform the data
df = pd.DataFrame(imputed_data, columns=df.columns)


# Features (X) and target variable (y)
df['target'] = np.log(df['target'])

# Split the data into training and testing sets
X_train, y_train, X_val, y_val, X_test, y_test = split_data(df, val_ratio=0.1, test_ratio=0.1, random_state=42)

# Parameter Space for Optuna to search and optimize
hyperparameters = {
        'learning_rate': lambda trial: trial.suggest_float('learning_rate', 0.01, 0.3),
        'max_depth': lambda trial: trial.suggest_int('max_depth', 1, 11),
        'min_child_weight': lambda trial: trial.suggest_int('min_child_weight', 1, 11),
        'subsample': lambda trial: trial.suggest_float('subsample', 0.1, 1.0),
        'colsample_bytree': lambda trial: trial.suggest_float('colsample_bytree', 0.1, 1.0),
        # 'n_estimators': lambda trial: trial.suggest_int('n_estimators', 100, 1600),
        'gamma': lambda trial: trial.suggest_float('gamma', 0.0, 1.0),
    }

# Fixed parameters for the model (no hyperparameters to optimize)
fixed_params = {
    'objective': 'reg:squarederror',
    # 'huber_slope': 5,
    'tree_method': 'hist',
    'seed': 42,
    'device': 'cuda',
    'n_estimators':4000
}

# Initialize the OptunaObjective
hyperparameter_optimization = OptunaObjective(
    model_class=MODEL,
    metric=mean_absolute_percentage_error,
    hyperparam_space=hyperparameters,
    fixed_params_space=fixed_params,
    random_state=RANDOM_STATE,
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    validation_strategy= 'xgboost_val',
    cv_folds=5,
    n_trials=N_TRIALS,
    optimization_direction='minimize',
    return_model=True
)

# Perform hyperparameter optimization
best_params, best_value, best_model, study = hyperparameter_optimization.optimize(show_progress_bar=True)

# Save the best model
best_model.save_model(MODEL_SAVE_NAME)

# Print the best hyperparameters and the best value
print("Best trial:")
trial = study.best_trial
print("Metric: ", trial.value)
print("Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")