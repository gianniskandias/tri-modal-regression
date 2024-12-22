# Import necessary libraries
from autogluon.tabular import TabularDataset, TabularPredictor
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


# User defined variables
DATA_NAME = 'data_duplicate_removed'  # name of the csv file
TIME_LIMIT = 1800  # Time in seconda that the hyperparameter optimization will run
SAVE_MODEL_NAME = "autogluon_regressor_model_ln_3600"

# Load the dataset
df = pd.read_csv(f'./description/{DATA_NAME}.csv')

target_column = "target"  # Replace with the name of your target column
df[target_column] = np.log(df[target_column])

# Split the dataset into training and testing sets
train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

# Display basic information about the data
print("Training data shape:", train_data.shape)
print("Testing data shape:", test_data.shape)

# Train the AutoGluon regressor
# AutoGluon doesn't have random state for reproducibility
# AutoGluon will automatically optimize hyperparameters and models (or ensemble of models)
predictor = TabularPredictor(label=target_column, 
                             problem_type="regression",
                             ).fit(train_data, 
                                    time_limit=TIME_LIMIT,  
                                    presets="best_quality",  # Use a higher quality preset (longer training)
    )

# Evaluate the predictor on the test data
performance = predictor.evaluate(test_data)

# Print performance metrics
print("Performance metrics:")
print(performance)

# Save the trained model
if SAVE_MODEL_NAME:
    predictor.save(SAVE_MODEL_NAME)