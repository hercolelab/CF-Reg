import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset
import numpy as np
from typing import Tuple
import os

def get_dataset(name: str, processing: str = "norm") -> Tuple[TensorDataset, TensorDataset]:
    
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    scaler = StandardScaler() if processing == "standard" else MinMaxScaler()
    dtype = torch.float32
    seed = 42

    if name == "water":
        
        try:
            df=pd.read_csv('data/water_potability.csv')
            
        except Exception:
            
            raise ValueError(f"Water dataset is not inside the data folder! cwd {os.getcwd()}")
        
        df['ph'] = df['ph'].fillna(value=df['ph'].median())
        df['Sulfate'] = df['Sulfate'].fillna(value=df['Sulfate'].median())
        df['Trihalomethanes'] = df['Trihalomethanes'].fillna(value=df['Trihalomethanes'].median())
        
        X = df.drop('Potability',axis=1).values
        y = df['Potability'].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)

        scaler.fit(X_train)
        
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        
        train_set = TensorDataset(torch.tensor(X_train, dtype=dtype), torch.tensor(y_train,dtype=torch.long))
        test_set = TensorDataset(torch.tensor(X_test, dtype=dtype), torch.tensor(y_test, dtype=torch.long))
        
        return train_set, test_set
    
    elif name == "mnist":
        from torchvision import datasets
        from torchvision import transforms
        
        training_data =   datasets.MNIST("data", train=True, download=True)

        test_data = datasets.MNIST('data', train=False, download=True)
        
        raw_train_data = training_data.data.type(torch.float).unsqueeze(1)
        raw_test_data = test_data.data.type(torch.float).unsqueeze(1)
        
        train_normalized = (raw_train_data - 0) / (255)
        test_normalized = (raw_test_data - 0) / (255)

                
        train_set = TensorDataset(train_normalized, training_data.targets)
        test_set = TensorDataset(test_normalized, test_data.targets)

        return train_set, test_set
    
    elif name == "fashion":
        
        from torchvision import datasets
        from torchvision import transforms
        
        training_data =   datasets.FashionMNIST("data", train=True, download=True)

        test_data = datasets.FashionMNIST('data', train=False, download=True)
        
        train_set = TensorDataset(training_data.data.type(torch.float).unsqueeze(1), training_data.targets)
        test_set = TensorDataset(test_data.data.type(torch.float).unsqueeze(1), test_data.targets)

        return train_set, test_set
    
    elif name == "cifar10":
        
        from torchvision import datasets
        from torchvision import transforms
        
        training_data =   datasets.CIFAR10("data", train=True, download=True)

        test_data = datasets.CIFAR10('data', train=False, download=True)
        
        train_set = TensorDataset(torch.Tensor(training_data.data).type(torch.float16).permute(0,3,1,2), torch.Tensor(training_data.targets).type(torch.uint8))
        test_set = TensorDataset(torch.from_numpy(test_data.data).type(torch.float16).permute(0,3,1,2), torch.Tensor(test_data.targets).type(torch.uint8))

        return train_set, test_set
        
        
    else:
        raise ValueError(f"Dataset {name} is not available!")
            
class UnsqueezeTransform:
    def __init__(self, dim):
        self.dim = dim
    
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        return img.unsqueeze(self.dim) 
    
if __name__ == "__main__":
    
    
    train, test = get_dataset(name="mnist")