from torch.nn import Module
import torch
from  src.losses.losses import CounterfactualRegularizationLoss, DynamicCounterfactualRegularizationLoss

def get_loss(**kwargs) -> Module:
    
    name: str = kwargs.pop("type")
    
    if name == "regularized":
        """
        Takes as input alpha, a float number [0, +inf)
        """
        
        return CounterfactualRegularizationLoss(**kwargs)
    
    if name == "dyn_regularized":
        
        return DynamicCounterfactualRegularizationLoss()
    
    elif name == "normal":
        
        return  torch.nn.functional.cross_entropy
    
    else:
        
        raise ValueError(f"This loss has not been implemented yet!")