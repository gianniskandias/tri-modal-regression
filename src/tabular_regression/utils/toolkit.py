import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple

def split_data(
    df: pd.DataFrame, 
    train_ratio: float = 0.8, 
    val_ratio: float = 0.1, 
    test_ratio: float = 0.1, 
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Splits the DataFrame into train, validation, and test sets based on provided ratios.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        train_ratio (float): Ratio of the training set. Defaults to 0.8.
        val_ratio (float): Ratio of the validation set. Defaults to 0.1.
        test_ratio (float): Ratio of the test set. Defaults to 0.1.
        random_state (int): Random seed for reproducibility. Defaults to 42.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]: 
            The train, validation, and test sets split into features and target variables, 
            as (train_X, train_y, val_X, val_y, test_X, test_y).
    """
    
    assert train_ratio + val_ratio + test_ratio == 1, "The sum of train_ratio, val_ratio, and test_ratio must be 1."

    # First split: train_val and test
    train_val, test = train_test_split(df, test_size=test_ratio, random_state=random_state)

    # Proportionally split train_val into train and validation sets
    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
    train, val = train_test_split(train_val, test_size=val_ratio_adjusted, random_state=random_state)
    
    train_X = train.drop('target', axis=1)
    train_y = train['target']
    
    val_X = val.drop('target', axis=1)
    val_y = val['target']
    
    test_X = test.drop('target', axis=1)
    test_y = test['target']

    return train_X, train_y, val_X, val_y, test_X, test_y