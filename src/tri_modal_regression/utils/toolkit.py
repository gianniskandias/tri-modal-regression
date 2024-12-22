import pandas as pd
import os
from fuzzywuzzy import fuzz
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.experimental import enable_iterative_imputer # It must be guven for the MICE imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from typing import List, Tuple, Dict, Union, Literal
from torch import nn
import torch
from sklearn.preprocessing import QuantileTransformer
import pickle


def dataframe_with_image_files(
    df: pd.DataFrame, 
    image_dir: str,
    similarity_column: str = "description"
) -> pd.DataFrame:
    """
    Associates image files with descriptions in a DataFrame based on similarity.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing a column with text descriptions.
    image_dir : str
        The directory path where image files are stored.
    similarity_column : str, optional
        The column name in the DataFrame containing text descriptions 
        to match with image files, by default "description".

    Returns
    -------
    pd.DataFrame
        A DataFrame with an added column 'image_filename' that maps each 
        description to the best-matching image file name.
    """
    
    image_files = os.listdir(image_dir)
    transformed_image_names = {
        filename: filename.split('.')[0].replace('_', ' ')
        for filename in image_files
    }

    description_to_image = {}
    
    # Find the best matching image for each description
    for description in df[similarity_column].unique():
        best_match = max(
            transformed_image_names.items(),
            key=lambda item: fuzz.ratio(description, item[1])
        )[0]
        description_to_image[description] = best_match
        
    df['image_filename'] = df[similarity_column].map(description_to_image)
    
    return df


def standardize_dataframe(
    df: pd.DataFrame, 
    column_exceptions: List[str] = ['target']
) -> pd.DataFrame:
    """
    Standardizes all numeric columns in a DataFrame, excluding NaNs.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to standardize.

    Returns
    -------
    pd.DataFrame
        The standardized DataFrame.
    """
    
    for column in df.select_dtypes(include='number').columns:
        if column not in column_exceptions:
            df[column] = (df[column] - df[column].mean(skipna=True)) / df[column].std(skipna=True)
            df[column] = df[column].fillna(np.nan)

    return df


def extract_and_convert_to_int(
    df: pd.DataFrame, 
    columns: List[str]
) -> pd.DataFrame:
    """
    Extracts numbers from specified columns in a DataFrame and converts them to integers.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    columns (List[str]): List of column names to process.

    Returns:
    pd.DataFrame: DataFrame with processed columns.
    """

    # Iterate over the specified columns
    for column in columns:
        # Extract the numbers from the column using regular expressions
        df[column] = df[column].str.extract(r'(\d+)')
        # Convert the extracted numbers to integers
        df[column] = df[column].astype(int)

    # Return the DataFrame with processed columns
    return df


def split_data(
    df: pd.DataFrame, 
    train_ratio: float = 0.8, 
    val_ratio: float = 0.1, 
    test_ratio: float = 0.1, 
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits the DataFrame into train, validation, and test sets based on provided ratios.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        train_ratio (float): Ratio of the training set. Defaults to 0.8.
        val_ratio (float): Ratio of the validation set. Defaults to 0.1.
        test_ratio (float): Ratio of the test set. Defaults to 0.1.
        random_state (int): Random seed for reproducibility. Defaults to 42.

    Returns:
        Splitted_data (pd.DataFrame, pd.DataFrame, pd.DataFrame): The train, validation, and test sets.
    """
    assert train_ratio + val_ratio + test_ratio == 1, "The sum of train_ratio, val_ratio, and test_ratio must be 1."

    # First split: train_val and test
    train_val, test = train_test_split(df, test_size=test_ratio, random_state=random_state)

    # Proportionally split train_val into train and validation sets
    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
    train, val = train_test_split(train_val, test_size=val_ratio_adjusted, random_state=random_state)

    return train, val, test


def MICE_impute(
    df: pd.DataFrame,
    max_iter: int = 15,
    random_state: int = 42
) -> pd.DataFrame:
    
    """
    Perform multiple imputation by chained equations (MICE) on the given DataFrame.
    
    It works best when the missing values missing missing completely at random (MCAR) 
    or Missing At Random (MAR).

    Parameters:
        df (pd.DataFrame): The input DataFrame to be imputed.
        max_iter (int): The maximum number of iterations to perform. Defaults to 15.
        random_state (int): The random seed for reproducibility. Defaults to 42.

    Returns:
        pd.DataFrame: The imputed DataFrame.
    """

    imputer = IterativeImputer(max_iter=max_iter, random_state=random_state)
    imputed_data = imputer.fit_transform(df)
    imputed_df = pd.DataFrame(imputed_data, columns=df.columns)
    
    return imputed_df

def SimpleImputer_impute(
    df: pd.DataFrame,
    **kwargs
) -> pd.DataFrame:
    """
    Impute missing values in numeric columns of a DataFrame using SimpleImputer.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing data with missing values.
        **kwargs: Additional keyword arguments to pass to the SimpleImputer.

    Returns:
        pd.DataFrame: The DataFrame with missing values imputed.
    """

    # Initialize the SimpleImputer with the provided keyword arguments
    imputer = SimpleImputer(**kwargs)
    
    # Select only the numeric columns from the DataFrame
    numeric_cols = df.select_dtypes(include=['number']).columns
    
    # Impute missing values in numeric columns
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    
    return df


def count_parameters(model) -> int:
    """Counts the total number of trainable parameters in a PyTorch model.

    Args:
        model (torch.nn.Module): The PyTorch model.

    Returns:
        int: The total number of trainable parameters. 

    """
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params


def custom_weight_decay(
    model: nn.Module, 
    weight_decay_rate: float, 
    weight_decay_modules: List[str], 
    keep_last_bias: bool = True
) -> List[Dict[str, Union[nn.Parameter, float]]]:
    """
    Applies custom weight decay to a model's parameters, optionally excluding the last bias parameters of specified modules.

    Args:
        model (nn.Module): The neural network model whose parameters will be considered for weight decay.
        weight_decay_rate (float): The rate at which weight decay is applied to the model's parameters.
        weight_decay_modules (List[str]): A list of module names for which weight decay should be applied.
        keep_last_bias (bool, optional): If True, the last bias parameter in each specified module will not have weight decay applied. Defaults to True.

    Returns:
        List[Dict[str, Union[nn.Parameter, float]]]: A list of parameter dictionaries to be used with an optimizer. Each dictionary specifies parameters and their respective weight decay rates.
    """
    
    def last_submodules_biases(
        model: nn.Module, 
        weight_decay_modules: List[str]
    ) -> List[str]:
        """
        Get the last bias parameters in each of the specified modules
        
        Args:
            model (nn.Module): Neural network model
            weight_decay_modules (List[str]): Modules to apply weight decay
        Returns:
            List[str]: Names of the last bias parameters in each module
        """
        
        sub_modules = []
        for name, param in model.named_parameters():
            is_bias = (
                    param.dim() == 1 and  # it's a bias 
                    name.endswith('.bias') and  # it's a bias parameter
                    # Check if it's the last bias in its module
                    any(module in name for module in weight_decay_modules)
                )

            if is_bias:
                sub_modules.append(name)

        temp = {}
        n = len("bias")
        
        # overwrite same modules based on keys so to keep the last bias
        for sub in sub_modules:
            temp[sub[:-(n+2)]] = sub

        last_biases = list(temp.values())
        
        return last_biases
    
    
    params_list = []  # Initialize list to store parameter configurations
    
    if keep_last_bias:
        # Get the last biases in the specified modules to exclude them from weight decay
        last_biases = last_submodules_biases(model, weight_decay_modules)
        
        for name, param in model.named_parameters():
            # Check if the parameter belongs to the specified modules and not in the last biases
            if any(module in name for module in weight_decay_modules) and name not in last_biases:
                params_list.append({
                    'params': param, 
                    'weight_decay': weight_decay_rate
                })
        else:
                # Parameters not in specified modules have no weight decay
                params_list.append({
                    'params': param, 
                    'weight_decay': 0.0
                })
    
    else:
        # Apply weight decay to all parameters in specified modules
        for name, param in model.named_parameters():
            if any(module in name for module in weight_decay_modules):
                params_list.append({
                    'params': param, 
                    'weight_decay': weight_decay_rate
                })
            else:
                # Parameters not in specified modules have no weight decay
                params_list.append({
                    'params': param, 
                    'weight_decay': 0.0
                })
    
    return params_list  # Return the list of parameter configurations



def entropy_regularization_loss(
    gating_probs: torch.Tensor, 
    reg_lambda: float = 0.01
) -> torch.Tensor:
    """
    Entropy-based regularization to encourage all experts to be utilized.
    
    Args:
        gating_probs (torch.Tensor): Batch of gating weights, shape [batch_size, num_experts].
        reg_lambda (float, optional): Regularization strength. Defaults to 0.01.
    
    Returns:
        torch.Tensor: Regularization loss (scalar).
    """
    # Mean probability for each expert across the batch
    mean_prob = torch.mean(gating_probs, dim=0)  # Shape: [num_experts]
    
    # Compute entropy-based regularization term
    reg_loss = torch.sum(mean_prob * torch.log(mean_prob + 1e-6))  # Add epsilon to avoid log(0)
    return -reg_lambda * reg_loss  # Negate because we want to maximize entropy


def sparsity_regularization_loss(
    gating_probs: torch.Tensor, 
    reg_lambda: float = 0.01
) -> torch.Tensor:
    """
    L1 regularization term to encourage sparse gating weights.

    Args:
        gating_probs (torch.Tensor): Batch of gating weights, shape [batch_size, num_experts].
        reg_lambda (float, optional): Regularization strength. Defaults to 0.01.

    Returns:
        torch.Tensor: Regularization loss (scalar).
    """
    
    # Sum the absolute value along the last dimension (rows)
    aggregate_prob = torch.sum(torch.abs(gating_probs), dim=-1)
    
    # Compute the L1 regularization term
    reg_loss = torch.mean(aggregate_prob)
    
    return reg_lambda * reg_loss


class LossWithEntropyRegularization(nn.Module):
    def __init__(self, loss_fn: nn.Module, reg_lambda: float = 0.01) -> None:
        """
        Initialize the LossWithEntropyRegularization module.

        Args:
            loss_fn (nn.Module): The loss function to be used.
            reg_lambda (float, optional): The regularization strength. Defaults to 0.01.
        """
        super().__init__()
        self.loss_fn = loss_fn
        self.reg_lambda = reg_lambda
        
    def forward(
        self, 
        predictions: torch.Tensor,  # Predictions from the model
        targets: torch.Tensor,  # Targets for the predictions
        gating_weights: torch.Tensor  # Gating weights for the MoE
    ) -> torch.Tensor:  
        """
        Compute the loss with entropy regularization.

        Args:
            predictions (torch.Tensor): Predictions from the model.
            targets (torch.Tensor): Targets for the predictions.
            gating_weights (torch.Tensor): Gating weights for the MoE.

        Returns:
            torch.Tensor: The total loss, including the entropy regularization term.
        """
        loss = self.loss_fn(predictions, targets)
        reg_loss = entropy_regularization_loss(gating_weights, self.reg_lambda)
        return loss + reg_loss


def quantile_transform(
    df: pd.DataFrame,
    fit_column: str,
    distribution_strategy: Literal['uniform', 'normal'] = "uniform",
    save_name_transformation: str = None,
) -> Tuple[QuantileTransformer, pd.DataFrame]:
    """
    Quantile transform a column in a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing the column to be transformed.
        fit_column (str): The name of the column to be transformed.
        distribution_strategy (Literal['uniform', 'normal'], optional): The distribution strategy to be used. Defaults to "uniform".
        save_name_transformation (str, optional): The name of the file to save the fitted transformer to. Defaults to None.

    For more information, please refer to the documentation for the `QuantileTransformer` class: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.QuantileTransformer.html
    
    Returns:
        returned (Tuple[QuantileTransformer, pd.DataFrame]): A tuple containing the fitted transformer and the transformed DataFrame.
    """
    
    # Create a QuantileTransformer object
    quantile_transformer = QuantileTransformer(output_distribution=distribution_strategy)

    # Fit and transform the data
    df[fit_column] = quantile_transformer.fit_transform(df[fit_column].values.reshape(-1, 1)).reshape(-1)

    # Save the fitted transformer to a file 
    if save_name_transformation:
        with open(f'{save_name_transformation}.pkl', 'wb') as file: 
            pickle.dump(quantile_transformer, file)
    
    return quantile_transformer, df

