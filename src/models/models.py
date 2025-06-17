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

class BLogisticRegression(nn.Module): 
    def __init__(self, **kwargs):
        super(BLogisticRegression, self).__init__()
        self.linear = nn.Linear(kwargs["input_dim"], 1) # binary classification

    def forward(self, x):
        return self.linear(x).squeeze(1)
    
    def linearize(self, data):
        """
        Computes the first-order Taylor expansion of the LogisticRegression for each element in a batch.
        For a linear model, the output is already the linearized version.

        Args:
            data (torch.Tensor): A batch of input tensors of shape [batch_size, input_dim].

        Returns:
            dict: A dictionary containing:
                - 'output': The output of the LogisticRegression for the batch, shape [batch_size, 1].
                - 'gradient': The model's parameters (weights), shape [1, input_dim].
                - 'linearized': The model itself, represented by the weights and bias.
        """
        # Compute the forward pass
        output = self.forward(data)  # Shape: [batch_size, 1]

        # Extract the model's parameters (weights and bias)
        weights = self.linear.weight  # Shape: [1, input_dim]
        #bias = self.linear.bias.detach()  # Shape: [1]

        return {
            "output": output,  # Raw model outputs
            "gradient": weights,  # Model's gradients w.r.t input are the weights
            "linearized": self,  # Linearized representation of the model
        }
        #TODO define a linearize function that linearize the metod at some datapoints, since LogisticRegression is linear, it doesn't do anything but agrees to the same output
        #       of MLP.linearize

class BMLP(nn.Module):
    def __init__(self, **kwargs):
        super(BMLP, self).__init__()
        self.layers = nn.ModuleList()
        self.use_dropout = kwargs["dropout"] > 0.0
        self.apply_softmax = kwargs.get("apply_softmax", False)  # Optional parameter
        
        # Create the first layer from the input dimension to the first hidden layer size
        current_dim = kwargs["input_dim"]
        for hidden_dim in kwargs["hidden_layers"]:
            self.layers.append(nn.Linear(current_dim, hidden_dim))
            if self.use_dropout:
                self.layers.append(nn.Dropout(kwargs["dropout"]))
            current_dim = hidden_dim
        
        # Output layer
        self.layers.append(nn.Linear(current_dim, 1))

    def forward(self, x: torch.Tensor):
        # Apply a ReLU activation function and dropout (if used) to each hidden layer
        for layer in self.layers[:-1]:
            x = layer(x)
            if isinstance(layer, nn.Linear):
                x = F.relu(x)

        # Output layer
        x = self.layers[-1](x).squeeze(1)
        
        # Apply softmax if required
        if self.apply_softmax:
            x = F.softmax(x, dim=-1)
        
        return x

    def linearize(self, x: torch.Tensor):
        """
        Computes the first-order Taylor expansion of the MLP for each element in a batch.

        Args:
            x (torch.Tensor): A batch of input tensors of shape [batch_size, input_dim].

        Returns:
            dict: A dictionary containing:
                - 'output': The output of the MLP for the batch, shape [batch_size, nclasses].
                - 'gradient': The gradient of the output w.r.t. the input, shape [batch_size, nclasses, input_dim].
                - 'linearized': The linearized approximation for each input in the batch, shape [batch_size, nclasses].
        """
    
        # Ensure gradients are tracked for the input
        x.requires_grad_(True)

        # Compute the forward pass
        output = self.forward(x)  # Shape: [batch_size, 1]
        #assert output.requires_grad #"Output tensor does not require gradients."
        # Compute the gradient of the output w.r.t. the input
        gradients = torch.autograd.grad(
            outputs=output,  # The output tensor
            inputs=x,        # The input tensor to differentiate with respect to
            grad_outputs=torch.ones_like(output),  # Gradient of the outputs
            create_graph=True,  # Keep computation graph for higher-order gradients
            retain_graph=True,  # Retain graph for repeated backward calls
        )[0]  # Shape: [batch_size, input_dim]

        # Compute the linearized approximation using Taylor expansion
        # Here we approximate the behavior of the model locally around x
        linearized_output = None  # it should be a function TODO

        return {
            "output": output,  # Original outputs
            "gradient": gradients,  # Gradients of the output w.r.t. the input
            "linearized": linearized_output,  # First-order approximation
        }
    

class MLP(nn.Module):
    def __init__(self, **kwargs):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.use_dropout = kwargs["dropout"] > 0.0
        self.apply_softmax = kwargs.get("apply_softmax", False)  # Optional parameter
        
        # Create the first layer from the input dimension to the first hidden layer size
        current_dim = kwargs["input_dim"]
        for hidden_dim in kwargs["hidden_layers"]:
            self.layers.append(nn.Linear(current_dim, hidden_dim))
            if self.use_dropout:
                self.layers.append(nn.Dropout(kwargs["dropout"]))
            current_dim = hidden_dim
        
        # Output layer
        self.layers.append(nn.Linear(current_dim, kwargs["nclasses"]))

    def forward(self, x: torch.Tensor):
        # Apply a ReLU activation function and dropout (if used) to each hidden layer
        for layer in self.layers[:-1]:
            x = layer(x)
            if isinstance(layer, nn.Linear):
                x = F.relu(x)
        
        # Output layer
        x = self.layers[-1](x)
        
        # Apply softmax if required
        if self.apply_softmax:
            x = F.softmax(x, dim=-1)
        
        return x

    def linearize(self, x: torch.Tensor):
        """
        Computes the first-order Taylor expansion of the MLP for each element in a batch.

        Args:
            x (torch.Tensor): A batch of input tensors of shape [batch_size, input_dim].

        Returns:
            dict: A dictionary containing:
                - 'output': The output of the MLP for the batch, shape [batch_size, nclasses].
                - 'gradient': The gradient of the output w.r.t. the input, shape [batch_size, nclasses, input_dim].
                - 'linearized': The linearized approximation for each input in the batch, shape [batch_size, nclasses].
        """
        print("Da implementare")
        exit()
        #    # Ensure gradients are tracked for the input
        #    x.requires_grad_(True)

        #    # Compute the forward pass and optionally apply softmax
        #    def forward_single_input(x_single):
        #        logits = self.forward(x_single.unsqueeze(0)).squeeze(0)
        #        if self.apply_softmax:
        #            return F.softmax(logits, dim=-1)  # Apply softmax to logits
        #        return logits  # Raw logits if softmax is not applied

        #    # Compute the Jacobian for each input in the batch
        #    # Shape of jacobian: [batch_size, nclasses, input_dim]
        #    jacobian = torch.autograd.functional.jacobian(
        #        forward_single_input, x, create_graph=True
        #    )

        #    # Compute the forward pass
        #    output = self.forward(x)  # Shape: [batch_size, nclasses]

        #    # Apply softmax to the output if needed
        #    if self.apply_softmax:
        #        output = F.softmax(output, dim=-1)

        #    # Compute the linearized approximation
        #    # Delta x: perturbation around the input
        #    delta_x = x - x.detach()
        #    linearized_approximation = output + torch.einsum("bki,bi->bk", jacobian, delta_x)   #TODO this line contains an error

        #    return {
        #        "output": output.detach(),  # Original outputs
        #        "gradient": jacobian.detach(),  # Gradients for each input-output pair
        #        "linearized": linearized_approximation.detach(),  # First-order approximation
        #    }


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
    
class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(x) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out

class BPreActResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, channel_in = 3):
        super(BPreActResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(channel_in, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.bn = nn.BatchNorm2d(512 * block.expansion)
        self.linear = nn.Linear(512 * block.expansion, 1)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.relu(self.bn(out))
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out).squeeze(1)
        return out
    
    def linearize(self, x: torch.Tensor):
        """
        Computes the first-order Taylor expansion of the MLP for each element in a batch.

        Args:
            x (torch.Tensor): A batch of input tensors of shape [batch_size, input_dim].

        Returns:
            dict: A dictionary containing:
                - 'output': The output of the MLP for the batch, shape [batch_size, nclasses].
                - 'gradient': The gradient of the output w.r.t. the input, shape [batch_size, nclasses, input_dim].
                - 'linearized': The linearized approximation for each input in the batch, shape [batch_size, nclasses].
        """
    
        # Ensure gradients are tracked for the input
        x.requires_grad_(True)

        # Compute the forward pass
        output = self.forward(x)  # Shape: [batch_size, 1]
        #assert output.requires_grad #"Output tensor does not require gradients."
        # Compute the gradient of the output w.r.t. the input
        gradients = torch.autograd.grad(
            outputs=output,  # The output tensor
            inputs=x,        # The input tensor to differentiate with respect to
            grad_outputs=torch.ones_like(output),  # Gradient of the outputs
            create_graph=True,  # Keep computation graph for higher-order gradients
            retain_graph=True,  # Retain graph for repeated backward calls
        )[0]  # Shape: [batch_size, input_dim]
        
        # Compute the linearized approximation using Taylor expansion
        # Here we approximate the behavior of the model locally around x
        linearized_output = None  # it should be a function TODO

        return {
            "output": output,  # Original outputs
            "gradient": gradients,  # Gradients of the output w.r.t. the input
            "linearized": linearized_output,  # First-order approximation
        }

class BCNN(nn.Module):
    
    def __init__(self, dimension_input: int, classes: int, channel_input: int, channel_list: list[int], kernel_list: list[int]):
        super(BCNN, self).__init__()
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
            
        self.layers.append(nn.Linear(channel_list[-1]*self.shapes[-1]*self.shapes[-1], 1))


    def forward(self, x: torch.Tensor):
        
        for layer in self.layers[:-1]:
            x = layer(x)
            
            if isinstance(layer, nn.MaxPool2d):
                x = F.relu(x)
                
        x = torch.flatten(x, start_dim=1)
        x = self.layers[-1](x).squeeze(1)
        return x
    
    def linearize(self, x: torch.Tensor):
        """
        Computes the first-order Taylor expansion of the MLP for each element in a batch.

        Args:
            x (torch.Tensor): A batch of input tensors of shape [batch_size, input_dim].

        Returns:
            dict: A dictionary containing:
                - 'output': The output of the MLP for the batch, shape [batch_size, nclasses].
                - 'gradient': The gradient of the output w.r.t. the input, shape [batch_size, nclasses, input_dim].
                - 'linearized': The linearized approximation for each input in the batch, shape [batch_size, nclasses].
        """
    
        # Ensure gradients are tracked for the input
        x.requires_grad_(True)

        # Compute the forward pass
        output = self.forward(x)  # Shape: [batch_size, 1]
        #assert output.requires_grad #"Output tensor does not require gradients."
        # Compute the gradient of the output w.r.t. the input
        gradients = torch.autograd.grad(
            outputs=output,  # The output tensor
            inputs=x,        # The input tensor to differentiate with respect to
            grad_outputs=torch.ones_like(output),  # Gradient of the outputs
            create_graph=True,  # Keep computation graph for higher-order gradients
            retain_graph=True,  # Retain graph for repeated backward calls
        )[0]  # Shape: [batch_size, input_dim]
        
        # Compute the linearized approximation using Taylor expansion
        # Here we approximate the behavior of the model locally around x
        linearized_output = None  # it should be a function TODO

        return {
            "output": output,  # Original outputs
            "gradient": gradients,  # Gradients of the output w.r.t. the input
            "linearized": linearized_output,  # First-order approximation
        }
    
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
        
    