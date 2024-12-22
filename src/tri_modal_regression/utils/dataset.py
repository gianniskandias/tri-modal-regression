import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from transformers import BertTokenizer, BertModel
from pathlib import Path
import numpy as np
from typing import Tuple
import warnings

# Ignore warnings from pandas that are not relevant and correct in this case
warnings.simplefilter(action='ignore', category=FutureWarning)

# Ignore warnings from opening a palette image and making it RGB directly (quality hasn't changed)
warnings.filterwarnings("ignore", category=UserWarning)


class CustomDataset(Dataset):
    def __init__(self, 
                 df: pd.DataFrame, 
                 image_dir: str, 
                 tokenizer_name: str = "google/bert_uncased_L-4_H-256_A-4", # smallest BERT
                 tokenizer_device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 text_column: str = "description",
                 image_column: str = "image_filename",
                 target_column: str = "target",
                 exponential_target: bool = True,
                 image_channels_type: str = 'RGB'
        ) -> None:
        """
        Initialize the CustomDataset class.

        Parameters
        ----------
        df : pd.DataFrame
            The dataframe containing the data.
        image_dir : str
            The directory where the images are stored.
        tokenizer_name : str, optional
            The name of the tokenizer model to use. Defaults to "google/bert_uncased_L-4_H-256_A-4".
        tokenizer_device : str, optional
            The device that the tokenizer will be loaded on. Defaults to "cuda" if available, otherwise "cpu".
        text_column : str, optional
            The name of the column containing the text data. Defaults to "description".
        image_column : str, optional
            The name of the column containing the image filenames. Defaults to "image_filename".
        target_column : str, optional
            The name of the column containing the target values. Defaults to "target".
        exponential_target : bool, optional
            Whether to apply an exponential transformation to the target values. Defaults to True.
        image_channels_type : str, optional
            The type of the image channels. Defaults to 'RGB'.
        """
        
        
        # Tabular 
        self.tabular_data = df.drop(columns=[text_column, image_column, target_column])
        
        # Target
        target = torch.tensor(df[target_column].values)
        self.target = target if not exponential_target else torch.log(target)
        
        # Text 
        self.text_column_name = text_column
        self.text_data = df[text_column]
        self.device = tokenizer_device
        
        # Text transformations
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name) # Text tokenizer (using HuggingFace tokenizer, e.g., BERT)
        self.model =  BertModel.from_pretrained(tokenizer_name).to(self.device) #TODO remove it
        self.max_length = self._get_max_token_length() # max length of text
        
        # Image
        self.image_column = Path(image_dir) / df[image_column]
        self.image_channels_type = image_channels_type
        
        # Image transformations
        transformation = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])
            
        images_mean, images_std = self._image_normalize(transformation)
        self.image_transforms = transforms.Compose(
            transformation.transforms + [
                transforms.Normalize(mean=images_mean, std=images_std)  # Normalize to [-1, 1]]
                ]
        )
        
        
    def _image_normalize(self, 
                         transformation: torch.nn.Module
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the mean and standard deviation of the image values in the dataset.
        
        This function is used to normalize the image values to zero mean and unit variance.
        
        Parameters
        ----------
        transformation : torch.nn.Module
            A transformation to be applied to the images before computing the statistics.
        
        Returns
        -------
        mean : torch.Tensor
            The mean of the image values.
        std : torch.Tensor
            The standard deviation of the image values.
        """
        
        mean = torch.zeros(3)
        std = torch.zeros(3)
        
        images = self.image_column.unique()
        for image in images:
            image = Image.open(image).convert(self.image_channels_type)
            image = transformation(image)
            
            mean += torch.mean(image, dim=[1, 2]) # 0:channel, 1:height, 2:width -> mean over height and width in each channel
            std += torch.std(image, dim=[1, 2])
            
        mean /= len(images)
        std /= len(images)
        
        return mean, std

    def _get_max_token_length(self) -> int:
        """
        Get the maximum length of the tokenized texts in the data frame
        
        Parameters
        ----------
        df : pandas.DataFrame
            The data frame containing the text data
            
        Returns
        -------
        int
            The maximum length of the tokenized texts
        """
        
        max_length = 0
        # Iterate over the text data in the data frame
        for text in self.text_data:
            # Tokenize the text
            tokenized_text = self.tokenizer(text, truncation=True, padding='max_length', max_length=512)
            # Update the maximum length if necessary
            max_length = max(max_length, np.count_nonzero(tokenized_text['input_ids']))

        return max_length

    def _encode_texts(self, idx) -> torch.Tensor:
        """
        Encode a text samples into a BERT embedding
        
        Parameters
        ----------
        idx : int
            The index of the text samples in the data frame
            
        Returns
        -------
        torch.Tensor
            The BERT embedding of the text sample with shape (max_length, hidden_size)
        """
        
        # Tokenize the text
        
        inputs = self.tokenizer(
            self.text_data.iloc[idx], 
            return_tensors='pt', 
            max_length=self.max_length, 
            padding='max_length', 
            truncation=True
        ).to(self.device)
            
        # Obtain the BERT embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state # (batch, max_length, hidden_size)
                
            
        return embeddings.squeeze(0)

    def __len__(self):
        return len(self.tabular_data)

    def __getitem__(self, idx) -> dict:
        """
        Get a single data point from the dataset.
        
        Parameters
        ----------
        idx : int
            Index of the data point to retrieve
            
        Returns
        -------
        dict
            A dictionary containing the tabular data, text data, image data, and target
        """
        
        # Get tabular data
        tabular_data = torch.tensor(self.tabular_data.iloc[idx])
        
        # Get text data and tokenize
        text = self._encode_texts(idx)
                
        # Get image data
        image = Image.open(self.image_column.iloc[idx]).convert(self.image_channels_type)
        image = self.image_transforms(image)
        
        # Target
        target = self.target[idx]
        
        return {
            'tabular_data': tabular_data,
            'text_data': text,
            'image': image,
            'target': target
        }
        
        