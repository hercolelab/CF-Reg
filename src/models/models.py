import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import *
from src.utility.geometric import Sphere
seed = 42





torch.manual_seed(seed)

# Additional steps to enforce determinism
# Note: These settings can degrade performance and may not guarantee complete reproducibility across different PyTorch releases or different platforms like CPUs and GPUs.

# Ensuring that all operations are deterministic on GPU (if using CUDA)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MLP(nn.Module):
    def __init__(self, ):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.use_dropout = dropout > 0.0
        
        # Create the first layer from the input dimension to the first hidden layer size
        current_dim = input_dim
        for hidden_dim in hidden_layers:
            self.layers.append(nn.Linear(current_dim, hidden_dim))
            if self.use_dropout:
                self.layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
        
        # Output layer
        self.layers.append(nn.Linear(current_dim, output_dim))
        
    def forward(self, x: torch.Tensor):
        # Apply a ReLU activation function and dropout (if used) to each hidden layer
        for layer in self.layers[:-1]:
            x = layer(x)
            if isinstance(layer, nn.Linear):
                x = F.relu(x)
        # No activation function for the output layer (assuming classification task)
        x = self.layers[-1](x)
        return x
    

def extract_embeddings_hook(module, input, output):
    
    module.embeddings = output
    
    
class CNN(nn.Module):
    
    def __init__(self, dimension_input: int, classes: int, channel_input: int, channel_list: list[int], kernel_list: list[int]):
        super(CNN, self).__init__()
        self.shapes: list[int] = []
        self.layers: nn.ModuleList = nn.ModuleList()
        current_channel = channel_input
        
        for channels, kernel in zip(channel_list, kernel_list):
            self.layers.append(nn.Conv2d(in_channels=current_channel, out_channels=channels, kernel_size=kernel))
            self.shapes.append(self.output_shape(edge=dimension_input, kernel_size=kernel))
            dimension_input = self.shapes[-1]
            self.layers.append(nn.MaxPool2d(kernel_size=2))
            self.shapes.append(self.output_shape(edge=dimension_input, kernel_size=2, stride=2))
            current_channel = channels
            dimension_input = self.shapes[-1]
            
        self.layers.append(nn.Linear(channel_list[-1]*self.shapes[-1]*self.shapes[-1], classes))


    def forward(self, x: torch.Tensor):
        
        for layer in self.layers[:-1]:
            x = layer(x)
            
            if isinstance(layer, nn.MaxPool2d):
                x = F.relu(x)
                
        x = torch.flatten(x, start_dim=1)
        x = self.layers[-1](x)
        return F.log_softmax(x, dim=-1)
    
    def output_shape(self, edge: int, kernel_size: int = 1, stride: int = 1, padding: int = 0, dilation: int = 1) -> int:
        
        out = (edge + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
        return out
    



class NoiseModule(nn.Module):
    
    def __init__(self, 
                 shape: Tuple[int, ...],
                 distribution: str = "normal", 
                 n_samples: int = 10, 
                 radius: float = 1.0,
                 *args, **kwargs) -> None:
        
        super().__init__(*args, **kwargs)
        
        self.__sphere: Sphere = Sphere
        self.__random_function = self.__sphere.random_normal_points_in_sphere if distribution == "normal" else self.__sphere.random_uniform_points_in_sphere
        self.__perturbation = self.__random_function(num_points=n_samples, shape=shape, radius=radius)
        self.__n_samples = n_samples
        self.__shape = shape
        
    def forward(self, x: Tensor):
        
        batch_size: int = x.shape[0]
        unit_dims: Tuple[int, ...] = (1, ) 
        new_shape: Tuple[int, ...] = (self.__n_samples, *unit_dims, *self.__shape)
        perturbation: Tensor = self.__perturbation.view(new_shape)
        repeat_dims: Tuple[int, ...] = (1, batch_size, *((1, )*len(new_shape[2:])))
        perturbation: Tensor = perturbation.repeat(repeat_dims)       
        data: Tensor = data.to(device=self.device) 
        sample_perturbed: Tensor = x + perturbation 
        batch_dims: Tuple[int, ...] = (-1, *new_shape[2:])
        sample_perturbed: Tensor = sample_perturbed.reshape(batch_dims)
        
        return sample_perturbed
        