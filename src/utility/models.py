import torch


def get_model(**kwargs) -> torch.nn.Module:
    
    
    config: dict = kwargs["config"]
    model_type: str = config.pop("model_type")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    
    if model_type == "MLP":
        from src.models.models import MLP
        
        model = MLP(**config)
        
        return model.to(device)
    
    elif model_type == "CNN":
        from src.models.models import CNN
        
        model = CNN(dimension_input=config["input_dim"][0], 
                    classes=config["nclasses"], 
                    channel_input=config["channel_in"], 
                    channel_list=config["channel_list"], 
                    kernel_list=config["kernel_list"],
                    n_samples=config["n_samples"],
                    radius=config["radius"])
        
        return model.to(device)
    
    else:
        
        raise ValueError(f"{model_type} is not a valide model type!")

    