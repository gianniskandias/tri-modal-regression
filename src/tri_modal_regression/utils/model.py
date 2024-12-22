from pytorch_tabnet.tab_network import TabNetEncoder
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal, Optional, Tuple, Union


torch.manual_seed(42)
torch.set_default_dtype(torch.float64)


class MultiModalBase(nn.Module):
    
    def __init__(
        self,
        n_tabular_features: int,
        n_steps: int = 4,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        """
        Initialize the multimodal base model. It returns the three modalities feature embeddings.

        Parameters
        ----------
        n_tabular_features : int
            The number of features in the tabular data.
        n_steps : int, optional
            The number of steps in the TabNet encoder. Defaults to 4.
        device : str, optional
            The device to use for training. Defaults to "cuda" if available, otherwise "cpu".
        """
        super().__init__()

        self.tabnet_encoder = TabNetEncoder(n_tabular_features, 1, n_steps=n_steps) # 1 output for regression
        self.image_encoder = CNNModel()
        self.text_encoder = BiLSTM()

        # Move the group attention matrix to the specified device (bug from the original TabNet implementation)
        self.tabnet_encoder.group_attention_matrix = self.tabnet_encoder.group_attention_matrix.to(device)
        
        
    def forward(
        self,
        tabular: torch.Tensor,
        text: torch.Tensor,
        image: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the multimodal base model. It returns the three modalities feature embeddings.

        Parameters
        ----------
        tabular : torch.Tensor
            The input tabular data.
        text : torch.Tensor
            The input text data.
        image : torch.Tensor
            The input image data.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            A tuple containing the aggregated feature embeddings of the tabular data,
            the feature embedding of the text data, and the feature embedding of the image data.
        """
        tabular_embedding, _ = self.tabnet_encoder(tabular)
        text_embedding = self.text_encoder(text)
        image_embedding = self.image_encoder(image)

        # Aggregating the feature embeddings of tabnet (step aggegation)
        # tabular_aggregated = torch.sum(torch.stack(tabular_embedding, dim=0), dim=0)
        tabular_aggregated = torch.cat(tabular_embedding, dim=1)

        return tabular_aggregated, text_embedding, image_embedding
    

class MultiModalAttentionRegressor(nn.Module):
    def __init__(self,
                 n_tabular_features: int,
                 n_steps: int = 4,
                 fusion_strategy: Literal['combined_modal_attention', 'gated_attention', 'modal_attention', None] = 'modal_attention',
                 tabular_embedding_dim: Optional[Union[torch.Tensor, int]] = None,
                 image_embedding_dim: Optional[Union[torch.Tensor, int]] = None,
                 text_embedding_dim: Optional[Union[torch.Tensor, int]] = None,
                 fusion_dim: Optional[Union[torch.Tensor, int]] = None,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
        ) -> None:
        """
        Initialize the multimodal regression model.

        Parameters:
            n_tabular_features (int): The number of features in the tabular data.
            n_steps (int): The number of steps in the TabNet encoder. Defaults to 4.
            fusion_strategy (Literal['combined_modal_attention', 'gated_attention', 'modal_attention', None]): The strategy to use for fusing the outputs from the three encoders. Defaults to 'modal_attention'.
            tabular_embedding_dim (Optional[int]): The dimension of the tabular data embedding. Defaults to None.
            image_embedding_dim (Optional[int]): The dimension of the image data embedding. Defaults to None.
            text_embedding_dim (Optional[int]): The dimension of the text data embedding. Defaults to None.
            fusion_dim (Optional[int]): The dimension of the fused embedding. Defaults to None.
            device (str): The device to use for training. Defaults to 'cuda' if available, otherwise 'cpu'.
        """
        super().__init__()
        
        self.tabular_embedding_dim = tabular_embedding_dim
        self.text_embedding_dim = text_embedding_dim
        self.image_embedding_dim = image_embedding_dim
        self.fusion_dim = fusion_dim
        
        self.base = MultiModalBase(n_tabular_features, n_steps, device)
        
        if fusion_strategy is not None:
            if fusion_strategy == 'combined_modal_attention':
                assert fusion_dim is not None, "If the fusion strategy is combined_modal_attention, the fusion dimension must be provided"
                self.modal_importance = self._modal_importance(
                    fusion_strategy, tabular_embedding_dim, text_embedding_dim, image_embedding_dim, fusion_dim)
                self.fc_regressor = nn.Linear(fusion_dim, 1)
            else:
                self.modal_importance = self._modal_importance(
                    fusion_strategy, tabular_embedding_dim, text_embedding_dim, image_embedding_dim)
                self.fc_regressor = nn.Linear(
                    tabular_embedding_dim + image_embedding_dim + text_embedding_dim, 1)
        
        self.tanh = nn.Tanh()
        

    def forward(self, tabular: torch.Tensor, text: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the multimodal regression model. It returns the predicted value.

        Parameters
        ----------
        tabular : torch.Tensor
            The tabular data.
        text : torch.Tensor
            The text data.
        image : torch.Tensor
            The image data.

        Returns
        -------
        torch.Tensor
            The output of the model.
        """
        
        tabular_output, text_output, image_output = self.base(tabular, text, image)
        if self.modal_importance is not None:
            output = self.modal_importance(tabular_output, text_output, image_output)
        else:
            output = torch.cat([tabular_output, text_output, image_output], dim=1)
        output = self.fc_regressor(self.tanh(output))
        
        return output
    

    def _modal_importance(
        self,
        fusion_strategy: Literal["combined_modal_attention", "gated_attention", "modal_attention"],
        tabular_dim: int,
        text_dim: int,
        image_dim: int,
        fusion_dim: Optional[int] = None,
    ) -> nn.Module:
        """
        Returns a module that implements the chosen multimodality fusion strategy.

        Args:
            fusion_strategy: The chosen strategy for fusing the feature embeddings of the three modalities.
            tabular_dim: The dimension of the tabular feature embeddings.
            text_dim: The dimension of the text feature embeddings.
            image_dim: The dimension of the image feature embeddings.
            fusion_dim: The dimension of the fused feature embeddings. Defaults to None.

        Returns:
            A module that implements the chosen multimodality fusion strategy.
        """
        if fusion_strategy == "combined_modal_attention":
            return LearnedCombinedImportance(tabular_dim, text_dim, image_dim, fusion_dim)
        elif fusion_strategy == "gated_attention":
            return GatedAttentionLayer(tabular_dim, text_dim, image_dim)
        elif fusion_strategy == "modal_attention":
            return LearnedModalImportanceLayer()



# Define the CNN Model
class CNNModel(nn.Module):
    def __init__(
        self, 
        input_channels: int = 3,
        layer_1st_kernels: int = 32,
        layer_2nd_kernels: int = 64,
        layer_3rd_kernels: int = 128,
        stride: int = 1,
        padding: int = 1,
        kernel_size: int = 3
    ) -> None:
        """
        Initializes a convolutional neural network (CNN) model for image classification.

        Args:
            input_channels (int, optional): The number of input channels. Defaults to 3.
            layer_1st_kernels (int, optional): The number of kernels in the first convolutional layer. Defaults to 32.
            layer_2nd_kernels (int, optional): The number of kernels in the second convolutional layer. Defaults to 64.
            layer_3rd_kernels (int, optional): The number of kernels in the third convolutional layer. Defaults to 128.
            stride (int, optional): The stride of the convolutional layers. Defaults to 1.
            padding (int, optional): The padding of the convolutional layers. Defaults to 1.
            kernel_size (int, optional): The kernel size of the convolutional layers. Defaults to 3.

        Attributes:
            conv_layers (nn.Sequential): The sequential container of convolutional layers.
            flatten_layer (nn.Flatten): The layer to flatten the output tensor.
            global_avg_pool_layer (nn.AdaptiveAvgPool2d): The layer to perform global average pooling.
        """
        
        super(CNNModel, self).__init__()
        
        self.conv_layers = nn.Sequential(
            # First convolutional layer
            nn.Conv2d(
                in_channels=input_channels, 
                out_channels=layer_1st_kernels, 
                kernel_size=kernel_size, 
                stride=stride, 
                padding=padding
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Second convolutional layer
            nn.Conv2d(
                in_channels=layer_1st_kernels, 
                out_channels=layer_2nd_kernels, 
                kernel_size=kernel_size, 
                stride=stride, 
                padding=padding
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Third convolutional layer
            nn.Conv2d(
                in_channels=layer_2nd_kernels, 
                out_channels=layer_3rd_kernels, 
                kernel_size=kernel_size, 
                stride=stride, 
                padding=padding
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.flatten_layer = nn.Flatten()
        self.global_avg_pool_layer = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the convolutional neural network (CNN) model.

        Args:
            x (torch.Tensor): The input tensor with shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: The output tensor with shape (batch_size, -1) after flattening.
        """
        # Perform convolutional operations
        x = self.conv_layers(x)
        
        # Perform global average pooling
        x = self.global_avg_pool_layer(x)
        
        # Flatten the output tensor
        x = self.flatten_layer(x)  # Flatten the output
        
        return x


# Definition of BiLSTM
class BiLSTM(nn.Module):
    def __init__(
        self,
        lstm_hidden_dim: int = 32,
        lstm_layers: int = 2,
        embedding_dim: int = 256,
        bidirectional: bool = True,
    ) -> None:
        """Initialize the BiLSTM model.

        Args:
            lstm_hidden_dim (int): The hidden dimension of the Bi-LSTM layer.
            lstm_layers (int): The number of layers in the Bi-LSTM layer.
            embedding_dim (int): The dimension of the input embeddings.
            bidirectional (bool): Whether the Bi-LSTM layer is bidirectional.
        """
        super().__init__()

        # Bi-LSTM layer
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            bidirectional=bidirectional,
            batch_first=True,
        )


    def forward(self, input_embeddings: torch.Tensor) -> torch.Tensor:
        """Forward pass through the Bi-LSTM layer.

        Args:
            input_embeddings (torch.Tensor): The input embeddings to the Bi-LSTM layer.

        Returns:
            torch.Tensor: The output of the Bi-LSTM layer.
        """
        # Pass input embeddings through Bi-LSTM
        output, _ = self.lstm(input_embeddings) # lstm_output shape: (batch_size, seq_length, 2*lstm_hidden_dim)

        # Average the output over the sequence dimension (mean pooling)
        output = torch.mean(output, dim=1) # (batch_size, 2*lstm_hidden_dim)

        return output


class GatedAttentionLayer(nn.Module):
    def __init__(self, tabular_dim: int, text_dim: int, image_dim: int):
        """
        Initialize the Gated Attention layer.

        Args:
            tabular_dim (int): Dimensionality of tabular features.
            text_dim (int): Dimensionality of text features.
            image_dim (int): Dimensionality of image features.
        """
        super().__init__()
        self.image_gate = nn.Linear(image_dim, 1)
        self.text_gate = nn.Linear(text_dim, 1)
        self.tabular_gate = nn.Linear(tabular_dim, 1)


    def forward(self, tabular_features: torch.Tensor, text_features: torch.Tensor, image_features: torch.Tensor) -> torch.Tensor:
        """
        Compute the gated attention output.

        Args:
            tabular_features (torch.Tensor): Features from the tabular data.
            text_features (torch.Tensor): Features from the text data.
            image_features (torch.Tensor): Features from the image data.

        Returns:
            torch.Tensor: The gated attention output.
        """
        # Compute gates [batch_size, 1]
        image_gate = torch.sigmoid(self.image_gate(image_features)) 
        text_gate = torch.sigmoid(self.text_gate(text_features))
        tabular_gate = torch.sigmoid(self.tabular_gate(tabular_features))

        # Normalize gates
        gate_sum = image_gate + text_gate + tabular_gate
        image_alpha = image_gate / gate_sum
        text_alpha = text_gate / gate_sum
        tabular_alpha = tabular_gate / gate_sum

        # Compute attended features
        attended_features = torch.cat([
            image_alpha * image_features,
            text_alpha * text_features,
            tabular_alpha * tabular_features
        ], dim=1)

        return attended_features


class LearnedModalImportanceLayer(nn.Module):
    def __init__(self, feature_dim: int = 3):
        """
        Initialize the learned modal importance layer. It weighs each modality by a factor
        
        Args:
            feature_dim (int): The number of features in the input tensor.
        """
        super().__init__()
        self.importance_weights = nn.Parameter(
            torch.ones(feature_dim), requires_grad=True
        )

    def forward(self, tabular_features: torch.Tensor, text_features: torch.Tensor, image_features: torch.Tensor) -> torch.Tensor:
        """
        Compute the weighted sum of the input features.

        Args:
            tabular_features: Features from the tabular data.
            text_features: Features from the text data.
            image_features: Features from the image data.

        Returns:
            torch.Tensor: The weighted sum of the input features.
        """
        # Compute softmax over learned weights
        weights = F.softmax(self.importance_weights, dim=0)  # [3]

        # Compute the weighted sum
        output = torch.cat([
            weights[0] * image_features,
            weights[1] * text_features,
            weights[2] * tabular_features,
        ], dim=1
        )

        return output


class LearnedCombinedImportance(nn.Module):
    def __init__(
        self,
        tabular_dim: torch.Tensor,
        text_dim: torch.Tensor,
        image_dim: torch.Tensor,
        attention_dim: int,
    ):
        """
        Initialize the learned combined importance layer.

        Args:
            tabular_dim (torch.Tensor): The number of features in the tabular data.
            text_dim (torch.Tensor): The number of features in the text data.
            image_dim (torch.Tensor): The number of features in the image data.
            attention_dim (int): The number of features in the attention space.
        """
        super().__init__()
        
        # Projection layers to map each modality's features to a common attention space
        self.proj_image = nn.Linear(image_dim, attention_dim)
        self.proj_text = nn.Linear(text_dim, attention_dim)
        self.proj_tabular = nn.Linear(tabular_dim, attention_dim)

        # Self-attention score layers for each modality
        self.score_self_image = nn.Linear(attention_dim, 1)
        self.score_self_text = nn.Linear(attention_dim, 1)
        self.score_self_tabular = nn.Linear(attention_dim, 1)

        # Cross-modality attention score layers
        self.score_image_text = nn.Linear(attention_dim, 1)
        self.score_text_tabular = nn.Linear(attention_dim, 1)
        self.score_tabular_image = nn.Linear(attention_dim, 1)

    def forward(
        self, 
        tabular_features: torch.Tensor, 
        text_features: torch.Tensor, 
        image_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            tabular_features (torch.Tensor): [batch_size, tabular_feature_dim]
            text_features (torch.Tensor): [batch_size, text_feature_dim]
            image_features (torch.Tensor): [batch_size, image_feature_dim]

        Returns:
            torch.Tensor: Attended features with shape [batch_size, feature_dim]
        """
        # Project features to a common attention space
        tabular_proj = torch.tanh(self.proj_tabular(tabular_features))
        text_proj = torch.tanh(self.proj_text(text_features))
        image_proj = torch.tanh(self.proj_image(image_features))

        # Self-attention scores
        tabular_self_score = self.score_self_tabular(tabular_proj)
        text_self_score = self.score_self_text(text_proj)
        image_self_score = self.score_self_image(image_proj)

        # Cross-modality attention scores
        tabular_image_score = self.score_tabular_image(tabular_proj * image_proj)
        text_tabular_score = self.score_text_tabular(text_proj * tabular_proj)
        image_text_score = self.score_image_text(image_proj * text_proj)

        # Concatenate attention scores
        attention_scores = torch.cat([
            tabular_self_score, text_self_score, image_self_score,
            tabular_image_score, text_tabular_score, image_text_score
        ], dim=1)

        # Normalize attention weights
        attention_weights = F.softmax(attention_scores, dim=1)

        # Weighted sums for each modality
        weighted_tabular_self = attention_weights[:, 0].unsqueeze(1) * tabular_proj
        weighted_text_self = attention_weights[:, 1].unsqueeze(1) * text_proj
        weighted_image_self = attention_weights[:, 2].unsqueeze(1) * image_proj

        weighted_tabular_image = attention_weights[:, 3].unsqueeze(1) * (tabular_proj + image_proj) / 2
        weighted_text_tabular = attention_weights[:, 4].unsqueeze(1) * (text_proj + tabular_proj) / 2
        weighted_image_text = attention_weights[:, 5].unsqueeze(1) * (image_proj + text_proj) / 2

        # Final attended features
        attended_features = (
            weighted_tabular_self +
            weighted_text_self +
            weighted_image_self +
            weighted_tabular_image +
            weighted_text_tabular +
            weighted_image_text
        )

        return attended_features


class MixtureOfExperts(nn.Module):
    def __init__(self, 
                 n_tabular_features: int,
                 n_steps: int = 4,
                 num_experts: int = 10,
                 hidden_dim: int = 64,
                 tabular_embedding_dim: Optional[Union[torch.Tensor, int]] = None, 
                 text_embedding_dim: Optional[Union[torch.Tensor, int]] = None, 
                 image_embedding_dim: Optional[Union[torch.Tensor, int]] = None, 
                 device:str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ):
        super().__init__()
        self.tabular_dim = tabular_embedding_dim
        self.text_dim = text_embedding_dim
        self.image_dim = image_embedding_dim
        self.input_dim = tabular_embedding_dim + text_embedding_dim + image_embedding_dim

        self.base = MultiModalBase(n_tabular_features, n_steps=n_steps, device=device)
        
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            ) for _ in range(num_experts)
        ])
        self.gating_network = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),  
            nn.Tanh(),
            nn.Linear(hidden_dim, num_experts),
            nn.Softmax(dim=1)
        )
        
        # TODO Change it if it delivers promising results, put it in forward for higher maintainability
        self.gate_weights = None # only for applying the entropy regularization

    def forward(self, tabular: torch.Tensor, text: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Mixture of Experts model.

        Args:
            tabular (torch.Tensor): The input tabular data.
            text (torch.Tensor): The input text data.
            image (torch.Tensor): The input image data.

        Returns:
            torch.Tensor: The final output of the model.
        """
        
        tabular_output, text_output, image_output = self.base(tabular, text, image)
        output = torch.cat([tabular_output, text_output, image_output], dim=1)
        output = F.tanh(output)  # [batch_size, output_dim]
        
        expert_outputs = torch.stack([expert(output) for expert in self.experts], dim=1)  # [batch_size, num_experts, output_dim]
        self.gate_weights = self.gating_network(output).unsqueeze(-1)  # [batch_size, num_experts, 1]
        
        weighted_outputs = expert_outputs * self.gate_weights
        final_output = weighted_outputs.sum(dim=1)  # [batch_size, output_dim]
        
        return final_output
