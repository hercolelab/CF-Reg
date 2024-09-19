import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import *
from src.utility.geometric import Sphere
seed = 42

torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MLP(nn.Module):
    
    def __init__(self, **kwargs):
        
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.use_dropout = kwargs["dropout"] > 0.0
        # Create the first layer from the input dimension to the first hidden layer size
        current_dim = kwargs["input_dim"]
        self.noise_module = NoiseModule(shape=(kwargs["hidden_layers"][-1],), n_samples=kwargs["n_samples"], radius=kwargs["radius"])

        for hidden_dim in kwargs["hidden_layers"]:
            self.layers.append(nn.Linear(current_dim, hidden_dim))
            if self.use_dropout:
                self.layers.append(nn.Dropout(kwargs["dropout"]))
            current_dim = hidden_dim
        
        # Output layer
        self.layers.append(nn.Linear(current_dim, kwargs["nclasses"]))
    
     
    def forward(self, x: torch.Tensor, use_noise_injection: bool = True):
        # Apply a ReLU activation function and dropout (if used) to each hidden layer
        for layer in self.layers[:-1]:
            x = layer(x)
            if isinstance(layer, nn.Linear):
                x = F.relu(x)
        # No activation function for the output layer (assuming classification task)
        if use_noise_injection:
            x=self.noise_module(x)
        x = self.layers[-1](x)
        return x
    

def extract_embeddings_hook(module, input, output):
    
    module.embeddings = output
    
    
class CNN(nn.Module):
    
    def __init__(self, dimension_input: int, classes: int, channel_input: int, channel_list: list[int], kernel_list: list[int], **kwargs):
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
        self.noise_module = NoiseModule(shape=(channel_list[-1]*self.shapes[-1]*self.shapes[-1],), n_samples=kwargs["n_samples"], radius=kwargs["radius"])
        self.layers.append(nn.Linear(channel_list[-1]*self.shapes[-1]*self.shapes[-1], classes))


    def forward(self, x: torch.Tensor, use_noise_injection: bool):
        
        for layer in self.layers[:-1]:
            x = layer(x)
            
            if isinstance(layer, nn.MaxPool2d):
                x = F.relu(x)
                
        x = torch.flatten(x, start_dim=1)
        if use_noise_injection:
            x = self.noise_module(x)
        x = self.layers[-1](x)
        return F.log_softmax(x, dim=-1)
    
    def output_shape(self, edge: int, kernel_size: int = 1, stride: int = 1, padding: int = 0, dilation: int = 1) -> int:
        
        out = (edge + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
        return out
    



class NoiseModule(nn.Module):
    
    def __init__(self, 
                 shape: Tuple[int, ...],
                 distribution: str = "uniform", 
                 n_samples: int = 10, 
                 radius: float = 1.0,
                 *args, **kwargs) -> None:
        
        super().__init__(*args, **kwargs)
        
        self.__sphere: Sphere = Sphere
        self.__random_function = self.__sphere.random_normal_points_in_sphere if distribution == "normal" else self.__sphere.random_uniform_points_in_sphere
        self.__perturbation = self.__random_function(num_points=n_samples, shape=shape, radius=radius)
        self.__n_samples = n_samples
        self.__shape = shape
        self.__device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def forward(self, data: Tensor):
        
        batch_size: int = data.shape[0]
        unit_dims: Tuple[int, ...] = (1, ) 
        new_shape: Tuple[int, ...] = (self.__n_samples, *unit_dims, *self.__shape)
        perturbation: Tensor = self.__perturbation.view(new_shape)
        repeat_dims: Tuple[int, ...] = (1, batch_size, *((1, )*len(new_shape[2:])))
        perturbation: Tensor = perturbation.repeat(repeat_dims)       
        data: Tensor = data.to(device=self.__device) 
        sample_perturbed: Tensor = data + perturbation 
        batch_dims: Tuple[int, ...] = (-1, *new_shape[2:])
        sample_perturbed: Tensor = sample_perturbed.reshape(batch_dims)
        #out: Tensor = self.function(sample_perturbed)
        
        return torch.cat((sample_perturbed, data))
        
        
