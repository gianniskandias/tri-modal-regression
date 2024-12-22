import pandas as pd
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import gc

# Hand crafted code
from utils import (SimpleImputer_impute, 
                    extract_and_convert_to_int, 
                    standardize_dataframe, 
                    dataframe_with_image_files,
                    split_data,
                    count_parameters,
                    custom_weight_decay,
                    quantile_transform
                    )
from utils import CustomDataset
from utils import MultiModalBase, MultiModalAttentionRegressor, MixtureOfExperts
from utils import MultiModalPipeline


# Reproducibility
torch.manual_seed(42)
torch.set_default_dtype(torch.float64)


IMAGE_DIR = './description/spacecraft_images'  # Directory where the image files are stored 
EPOCHS = 4
BATCH_SIZE = 128
N_STEPS = 8  # the number of steps in the TabNet encoder
MODEL_SAVED_NAME =  'gated_attention'  # name on how the model will be saved
EXPONENTIAL_TARGET = True  # log of the target variable
QUANTILE_TRANSFORM = False  # quantile transformation of the target variable
CONTINUE_TRAINING:str = None  # name of the model to continue training from
CRITERION = nn.MSELoss()

# Model on top of the modalities networks
MODEL_WITH_HEAD = MultiModalAttentionRegressor

# Specific to a simple head MLP on top of the fusion of the modalities
FUSION_STRATEGY = 'gated_attention'  # how the modalites (tabular, text, image) will be combined
FUSION_DIM = None  # dimension of the fused features if fusion_strategy is 'combined_modal_attention'

# Specific to the mixture of experts
ENTROPY_REGULARIZATION:float = None  # enforcing collaboration of experts (user value is 0.01)
SPARSITY_REGULARIZATION:float = None  # enforcing diversity of experts (user value is 0.01)
CUSTOMIZED_WEIGHT_DECAY:float = None  # Weight decay for experts only, except the last bias (user value is 0.001)


# Used Pipeline
def data_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the data by extracting and converting string features, dropping empty columns, 
    standardizing numeric columns, associating image files with descriptions, and imputing missing values."""
    
    # Extract and convert string features (feature_1 and featue_2) to numerical
    df = extract_and_convert_to_int(df, ['feature_1', 'feature_2'])

    # Drop completely empty columns (feature_12 and feature_17)
    df = df.drop(columns=['feature_12', 'feature_17'])

    # Standardize the numeric columns
    df = standardize_dataframe(df)

    # Associate image files with descriptions
    df = dataframe_with_image_files(df, image_dir=IMAGE_DIR, similarity_column='description')

    # Impute missing values
    df = SimpleImputer_impute(df, strategy='mean')
    
    return df


# Load the dataset
df = pd.read_csv('./description/candidates_data.csv')

# Preprocess the data
df = data_preprocessing(df)

# Split the dataset in a determinitic way
train, val, _ = split_data(df)

# Create a QuantileTransformer object
if QUANTILE_TRANSFORM:
    quantile_transformer, train = quantile_transform(train, 'target', save_name_transformation='./src/2nd_task/quantile_transformer/quantile_transformer')
    val['target'] = quantile_transformer.transform(val['target'].values.reshape(-1, 1)).reshape(-1)

# Create the datasets
train_dataset = CustomDataset(df=train, image_dir=IMAGE_DIR, exponential_target=EXPONENTIAL_TARGET)
val_dataset = CustomDataset(df=val, image_dir=IMAGE_DIR, exponential_target=EXPONENTIAL_TARGET)

# Create the dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)  # if device != 'cuda' on CustomDataset, put pin_memory=True (for efficiency)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Model Pipeline
model_pipeline = MultiModalPipeline()

# Build the implemented model (similar to build of TensorFlow)
model = model_pipeline.build(
    train_dataset=train_dataset, 
    base_model=MultiModalBase, 
    model_with_head=MODEL_WITH_HEAD,
    n_steps=N_STEPS,
    fusion_strategy= FUSION_STRATEGY,
    fusion_dim=FUSION_DIM,
    )


print(f"The total trainable parameters of the {model.__class__.__name__} are: {count_parameters(model)}\n")

# Model specifics
if CUSTOMIZED_WEIGHT_DECAY:
    optimizer = optim.AdamW(custom_weight_decay(
                                                model, 
                                                weight_decay_rate=5e-3, 
                                                weight_decay_modules=['gating_network', 'experts'], 
                                                keep_last_bias=True
                                                ), 
                            lr=0.001
                    )
else:
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.001)

scheduler = None  #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# if to continue training
if CONTINUE_TRAINING:
    checkpoint = torch.load(CONTINUE_TRAINING)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optim'])


gc.collect()
# Train the model
model_pipeline.train(
    model=model,
    loss_fn=CRITERION,
    optim=optimizer,
    scheduler=scheduler,
    train_loader=train_dataloader,
    val_loader=val_dataloader,
    num_epochs=EPOCHS,
    checkpoint_path= f'./{MODEL_SAVED_NAME}',
    entropy_regularization=ENTROPY_REGULARIZATION,
    sparsity_regularization=SPARSITY_REGULARIZATION
)
