from utils.model import MultiModalAttentionRegressor, MultiModalBase, MixtureOfExperts
from utils.toolkit import *
from utils.dataset import *
from typing import Literal, Optional
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch import nn
import warnings


torch.manual_seed(42)

class MultiModalPipeline:
    def __init__(self, model: nn.Module = None) -> None:
        """
        Initialize the MultiModalClass object with an optional model.

        Args:
            model (nn.Module, optional): The PyTorch model to be used. Defaults to None.
        """
        self.model = model
        
    def build(
        self,
        train_dataset: Dataset,
        base_model: nn.Module,
        model_with_head: nn.Module,
        n_steps: int = 4,
        fusion_strategy: Literal[
            'combined_modal_attention', 'gated_attention', 'modal_attention', None
        ] = 'modal_attention',
        fusion_dim: Optional[int] = None,
        **kwargs
    ) -> None:
        """
        Build the model from the given components. This method is similar to the build method of TensorFlow.

        Args:
            train_dataset: The dataset used for training.
            base_model: The base model used as the backbone.
            model_with_head: The model with a head to be used for training.
            fusion_strategy: The strategy to use for fusing the feature embeddings of the three modalities.
                Defaults to 'modal_attention'.
            n_steps: The number of steps in the TabNet encoder. Defaults to 4.
            fusion_dim: The dimension of the fused features. Defaults to None.

        Returns:
            None
        """
        # Check if a model already exists
        if self.model is not None:
            warnings.warn(
                "A model has already been built. The current instance will be overridden. Please create a new instance of MultiModalClass to build a new model."
            )

        # Extract the number of tabular features from the train dataset
        n_features = train_dataset[0]['tabular_data'].shape[0]  
        
        # Separation of kwargs
        prefix = 'sub_'
        kwargs1 = {k[len(prefix):]: v for k, v in kwargs.items() if k.startswith(prefix)} # currently not used
        kwargs2 = {k: v for k, v in kwargs.items() if not k.startswith(prefix)}

        # Initialize the base model with the specified number of features and steps
        model_backbone = base_model(n_tabular_features=n_features, n_steps=n_steps, device='cpu')

        # Create a sample input for the model
        sample_input = {
            'tabular_data': torch.randn(5, n_features),
            'text_data': torch.randn(
                5, train_dataset[0]['text_data'].shape[0], train_dataset[0]['text_data'].shape[1]
            ),
            'image': torch.randn(
                5,
                train_dataset[0]['image'].shape[0],
                train_dataset[0]['image'].shape[1],
                train_dataset[0]['image'].shape[2],
            ),
        }

        # Pass the sample input through the model backbone to obtain embeddings shapes
        tabular, text, image = model_backbone(
            sample_input['tabular_data'],
            sample_input['text_data'],
            sample_input['image'],
        )

        # Initialize the model with head using the obtained embeddings and fusion strategy
        if model_with_head is MultiModalAttentionRegressor:
            self.model = model_with_head(
                n_tabular_features=n_features,
                n_steps=n_steps,
                tabular_embedding_dim=tabular.shape[1],
                image_embedding_dim=image.shape[1],
                text_embedding_dim=text.shape[1],
                fusion_strategy=fusion_strategy,
                fusion_dim=fusion_dim,
                **kwargs2
            )
        elif model_with_head is MixtureOfExperts:
            self.model = model_with_head(
                n_tabular_features=n_features,
                n_steps=n_steps,
                tabular_embedding_dim=tabular.shape[1],
                image_embedding_dim=image.shape[1],
                text_embedding_dim=text.shape[1],
                **kwargs2
            )
        
        return self.model
        
    
    def train_epoch(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str,
        epoch: int,
        entropy_regularization: Union[None, float],
        sparsity_regularization: Union[None, float],
    ) -> float:
        """
        Train the model for one epoch.

        Args:
            model (nn.Module): The model to be trained.
            train_loader (DataLoader): DataLoader for the training data.
            loss_fn (nn.Module): The loss function.
            optimizer (torch.optim.Optimizer): The optimizer for training.
            device (str): The device to run the training on (e.g., 'cpu' or 'cuda').
            epoch (int): The current epoch number.
            entropy_regularization (Union[None, float]): The entropy regularization loss.
            sparsity_regularization (Union[None, float]): The sparsity regularization loss.

        Returns:
            float: The average loss over the epoch.
        """
        
        # Set the model to training mode
        model.train()

        # Initialize the total loss
        total_loss = 0.0

        # Create a progress bar with tqdm
        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}",
            ncols=140,
            colour="green",
        )

        # Iterate over the batches in the DataLoader
        for batch in progress_bar:
            # Zero the gradient of the optimizer
            optimizer.zero_grad()

            # Move the data to the specified device (e.g., 'cpu' or 'cuda')
            tabular = batch["tabular_data"].to(device)
            text = batch["text_data"].to(device)
            image = batch["image"].to(device)
            labels = batch["target"].to(device)

            # Unsqueeze the labels to match the shape of the outputs
            labels = labels.unsqueeze(1)

            # Forward pass
            outputs = model(tabular, text, image)

            # Calculate the loss
            if entropy_regularization:
                loss = loss_fn(outputs, labels) + entropy_regularization_loss(model.gate_weights, entropy_regularization)  # applied entropy_regularization = 0.01
            elif sparsity_regularization: #  L1 regularization
                loss = loss_fn(outputs, labels) + sparsity_regularization_loss(model.gate_weights, sparsity_regularization)  # applied sparsity_regularization = 0.01
            else:
                loss = loss_fn(outputs, labels)

            # Backward pass
            loss.backward()

            # Optimize the model parameters
            optimizer.step()

            # Add the loss to the total loss
            total_loss += loss.item()

            # Set the post-fix of the progress bar to the current loss
            progress_bar.set_postfix(loss=f"{total_loss / (progress_bar.n + 1):.4f}", batch_loss = f"{loss.item():.4f}")

        # Return the average loss over the epoch
        return total_loss / len(train_loader)
    
    
    def train(
        self, 
        model: nn.Module, 
        train_loader: DataLoader, 
        loss_fn: nn.Module, 
        optim: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        val_loader: Optional[DataLoader] = None, 
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu', 
        num_epochs: int = 2, 
        checkpoint_path: Optional[str] = None,
        entropy_regularization: Union[None, float] = None,
        sparsity_regularization: Union[None, float] = None
    ) -> None:
        """
        Train a PyTorch model on the given data.

        Args:
            model: The PyTorch model to be trained.
            train_loader: The DataLoader for the training data.
            val_loader: The DataLoader for the validation data. Defaults to None.
            loss_fn: The loss function to be used.
            optim: The optimizer to be used.
            scheduler: The learning rate scheduler to be used. Defaults to None.
            device: The device to be used for training (cpu or cuda). Defaults to 'cuda' if available.
            num_epochs: The number of epochs to be trained. Defaults to 2.
            checkpoint_path: The path to save the model checkpoint. Defaults to None.
            entropy_regularization: The entropy regularization parameter. Defaults to None.
            sparsity_regularization: The sparsity regularization parameter. Defaults to None.

        Returns:
            None
        """
        
        for epoch in range(1, num_epochs + 1):
            train_loss = self.train_epoch(model, train_loader, loss_fn, optim, device, epoch, entropy_regularization, sparsity_regularization)
    
            if val_loader:
                val_loss = self.evaluate(model, val_loader, loss_fn, device, epoch, num_epochs)
                print('-'*60)
                print(f'Epoch [{epoch}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
                print('='*60)
            
            # learning rate scheduler
            if scheduler:
                scheduler.step()
            
            
        if checkpoint_path:
            torch.save(
                {
                'model_state_dict': model.state_dict(), 
                'epoch': num_epochs,
                'optim': optim.state_dict(),
                'lr_scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'train_loss': train_loss,
                'val_loss': val_loss if val_loader else None,
            }, 
                f'{checkpoint_path}.pth'
            )
        
        print('='*60)
        print('Training complete')
            
    def evaluate(
        self, 
        model: nn.Module, 
        dataloader: DataLoader, 
        criterion: nn.Module, 
        device: str, 
        epoch: int, 
        total_epochs: int
    ) -> float:
        """
        Evaluate a PyTorch model on a given DataLoader.

        Args:
            model (nn.Module): The PyTorch model to be evaluated.
            dataloader (DataLoader): The DataLoader containing the evaluation data.
            criterion (nn.Module): The loss function to be used.
            device (str): The device to be used for evaluation (cpu or cuda).
            epoch (int): The current epoch number.
            total_epochs (int): The total number of epochs.

        Returns:
            float: The average loss over the evaluation data.
        """
        
        model.eval()
        total_loss = 0.0

        loader_tqdm = tqdm(
            iterable=dataloader, 
            desc=f'Epoch {epoch}/{total_epochs} [Validation]', 
            leave=True,
            unit='batch',
            ncols=140, 
            colour='yellow'
        )

        with torch.no_grad():
            for idx, batch in enumerate(loader_tqdm):
                tabular_data, text_data, image_data, target = (
                    batch['tabular_data'].to(device), 
                    batch['text_data'].to(device), 
                    batch['image'].to(device), 
                    batch['target'].to(device)
                )
                target = target.unsqueeze(1)
                output = model(tabular_data, text_data, image_data)
                loss = criterion(output, target)
                total_loss += loss.item()
                loader_tqdm.set_postfix(loss=total_loss / (idx + 1) , batch_loss = f"{loss.item():.4f}")

        return total_loss / len(dataloader)
        
        
    def predict(
        self, 
        model: nn.Module, 
        dataloader: DataLoader, 
        loss_fn: Optional[nn.Module] = None, 
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ) -> Tuple[Optional[float], torch.Tensor, torch.Tensor]:
        """
        Evaluate a PyTorch model on a given DataLoader.

        Args:
            model (nn.Module): The PyTorch model to be evaluated.
            dataloader (DataLoader): The DataLoader containing the evaluation data.
            loss_fn (Optional[nn.Module], optional): The loss function to be used. Defaults to None.
            device (str, optional): The device to be used for evaluation (cpu or cuda). Defaults depends on system.

        Returns:
            Optional[float]: The average loss over the evaluation data if loss_fn is provided, else None.
            torch.Tensor: The predictions.
            torch.Tensor: The true labels.
        """

        model.eval()
        total_loss = 0.0
        all_predictions: List[torch.Tensor] = []
        all_labels: List[torch.Tensor] = []

        dataloader_tqdm = tqdm(
            dataloader, desc='Evaluating', 
            unit='batch', 
            ncols=150, 
            colour='white',
            leave=True
        )

        with torch.no_grad():
            for batch in dataloader_tqdm:
                tabular, text, image, labels = (
                    batch['tabular_data'].to(device),
                    batch['text_data'].to(device),
                    batch['image'].to(device),
                    batch['target'].to(device)
                )
                labels = labels.unsqueeze(1)
                outputs = model(tabular, text, image)

                if loss_fn is not None:
                    loss = loss_fn(outputs, labels)
                    total_loss += loss.item()
                    dataloader_tqdm.set_postfix(loss=total_loss / (len(all_predictions) + 1))

                all_predictions.append(outputs)
                all_labels.append(labels)

        average_loss = total_loss / len(dataloader) if loss_fn is not None else None
        return average_loss, torch.cat(all_predictions), torch.cat(all_labels)
    
    