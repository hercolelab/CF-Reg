import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset
import numpy as np
from typing import Tuple
import os
from scipy.io import arff
import random

def features_transformation(X_train, X_test, preprocess_config):
    poly_features_enabled = preprocess_config['poly_features_enabled']
    rp_enabled = preprocess_config['rp_enabled']
    rff_enabled = preprocess_config['rff_enabled']

    if poly_features_enabled:
        from sklearn.preprocessing import PolynomialFeatures
        old_shape = X_train.shape
        poly = PolynomialFeatures(preprocess_config['poly_features_degree'])
        poly.fit(X_train)
        X_train = poly.transform(X_train)
        X_test = poly.transform(X_test)
        print(f"Polynomial Features of degree {preprocess_config['poly_features_degree']}. \nData from shape {old_shape} to shape {X_train.shape}.")
    
    elif rp_enabled:
        from sklearn.random_projection import GaussianRandomProjection
        if preprocess_config['rp_components_rate'] != 0:
            n_components = int(X_train.shape[0] * preprocess_config['rp_components_rate'])
        else:
            n_components = preprocess_config['rp_n_components']
        old_shape = X_train.shape
        rp = GaussianRandomProjection(n_components=n_components, random_state=42)
        rp.fit(X_train)
        X_train = rp.transform(X_train)
        X_test = rp.transform(X_test)
        print(f"Random Projection in {n_components}. \nData from shape {old_shape} to shape {X_train.shape}.")

    elif rff_enabled:
        from sklearn.kernel_approximation import RBFSampler
        if preprocess_config['rff_components_rate'] != 0:
            n_components = int(X_train.shape[0] * preprocess_config['rff_components_rate'])
        else:
            n_components = preprocess_config['rff_n_components']
        old_shape = X_train.shape
        rff = RBFSampler(n_components=n_components, random_state=42, gamma = "scale")
        rff.fit(X_train)
        X_train = rff.transform(X_train)
        X_test = rff.transform(X_test)
        print(f"Random Fourier Features in {n_components}. \nData from shape {old_shape} to shape {X_train.shape}.")
    
    else: 
        print("Preprocessing transformation technique not yet implemented")
        
    return X_train, X_test

def apply_noise(train_set, noise_rate, seed_noise, dtype_out):
    original_train_labels = train_set.tensors[1].clone()
    num_samples = len(original_train_labels)
    num_noisy_samples = int(noise_rate * num_samples)
    
    unique_classes = torch.unique(original_train_labels).tolist()
    if len(unique_classes) < 2:
        print("Warning: Cannot apply label noise with less than 2 unique classes.")
    else:
        random.seed(seed_noise)
        noisy_indices = random.sample(range(num_samples), num_noisy_samples)

        is_noisy_label = torch.zeros_like(original_train_labels, dtype=torch.bool)
        is_noisy_label[noisy_indices] = True

        for idx in noisy_indices:
            original_label = original_train_labels[idx].item()
            possible_flips = [c for c in unique_classes if c != original_label]
            if possible_flips:
                new_label = random.choice(possible_flips)
                train_set.tensors[1][idx] = torch.tensor(new_label, dtype=dtype_out)
        
        train_set = TensorDataset(train_set.tensors[0], train_set.tensors[1], original_train_labels, is_noisy_label)

        print(f"Applied {num_noisy_samples} ({noise_rate*100:.2f}%) noisy labels to the training set.")

    return train_set

def preprocess(df, preprocess_config, target_name):
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.utils import resample, shuffle
    
    seed_split = preprocess_config['seed_split']
    print("seed_split: ", seed_split)
    seed_resample = 42

    resample_value = preprocess_config['resample']
    if resample_value < 1:
        df = resample(df, n_samples=int(df.shape[0] * resample_value), random_state=seed_resample, stratify=df[target_name], replace=False)
    elif resample_value > 1:
        df = resample(df, n_samples=int(resample_value), random_state=seed_resample, stratify=df[target_name], replace=False)


    # Define features and target
    X = df.drop(target_name, axis=1).values
    y = df[target_name].values

    test_size = preprocess_config['test_size']
    # Split data into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed_split, stratify=df[target_name])

    if preprocess_config['scaler'] == "MinMax":
        scaler = MinMaxScaler()
    elif preprocess_config['scaler'] == "Standard":
        scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    #if (preprocess_config['poly_features_degree'] != 1):
    X_train, X_test = features_transformation(X_train, X_test, preprocess_config)
    
#    print(list(np.mean(X_train, axis = 0)))
#    print("\n","\n",list(np.std(X_train, axis = 0)))
    return X_train, X_test, y_train, y_test

def get_dataset(**kwargs) -> Tuple[TensorDataset, TensorDataset]:
    name: str = kwargs['name']
    binary_loss: bool = kwargs['binary']
    preprocess_config: dict = kwargs['preprocess_config']
    noise_rate: float = preprocess_config.pop('noise_rate', 0.0) # New parameter for noise rate
    seed_noise: int = preprocess_config.pop('seed_noise', 42) # Seed for reproducibility of noise



    dtype_in = torch.float32
    dtype_out = torch.float32 if binary_loss else torch.long
  

    if name == "water":
        
        try:
            df=pd.read_csv('data/water_potability.csv')
            
        except Exception:
            
            raise ValueError(f"Water dataset is not inside the data folder! cwd {os.getcwd()}")
        
        df['ph'] = df['ph'].fillna(value=df['ph'].median())
        df['Sulfate'] = df['Sulfate'].fillna(value=df['Sulfate'].median())
        df['Trihalomethanes'] = df['Trihalomethanes'].fillna(value=df['Trihalomethanes'].median())

        X_train, X_test, y_train, y_test = preprocess(df, preprocess_config, 'Potability')
        
        train_set = TensorDataset(torch.tensor(X_train, dtype=dtype_in), torch.tensor(y_train,dtype=dtype_out))
        test_set = TensorDataset(torch.tensor(X_test, dtype=dtype_in), torch.tensor(y_test, dtype=dtype_out))
        
        if noise_rate > 0:
            train_set = apply_noise(train_set, noise_rate, seed_noise, dtype_out)

        return train_set, test_set
    
    elif name == "mnist":
        from torchvision import datasets
        from torchvision import transforms


        
        training_data =   datasets.MNIST("data", train=True, download=True,
                             transform= transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ]))

        test_data = datasets.MNIST('data', train=False, download=True,
                             transform= transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ]))
        
        if binary_loss:
            # Select only instances belonging to classes 4 and 7
            train_mask = (training_data.targets == 4) | (training_data.targets == 7)
            test_mask = (test_data.targets == 4) | (test_data.targets == 7)

            training_data.data = training_data.data[train_mask]
            training_data.targets = training_data.targets[train_mask]
            test_data.data = test_data.data[test_mask]
            test_data.targets = test_data.targets[test_mask]

            # Relabel classes 4 -> 0 and 7 -> 1
            training_data.targets = (training_data.targets == 7).long()
            test_data.targets = (test_data.targets == 7).long()
            

        train_set = TensorDataset(training_data.data.type(torch.float).unsqueeze(1), torch.tensor(training_data.targets, dtype= dtype_out))
        test_set = TensorDataset(test_data.data.type(torch.float).unsqueeze(1), torch.tensor(test_data.targets, dtype= dtype_out))

        # Apply label noise to the training set for FashionMNIST
        if noise_rate > 0:
            train_set = apply_noise(train_set, noise_rate, seed_noise, dtype_out)


        #Print some data information
        labels = test_set.tensors[1]
        unique_classes, counts = torch.unique(labels, return_counts=True)
        total_samples = len(labels)
        proportions = counts / total_samples

        for cls, count, prop in zip(unique_classes, counts, proportions):
            print(f"Class {cls.item()}: Count = {count.item()}, Proportion = {prop.item():.2%}")

        return train_set, test_set
    
    elif name == "fashion":
        
        from torchvision import datasets
        from torchvision import transforms
        
        training_data =   datasets.FashionMNIST("data", train=True, download=True,
                             transform= transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ]))

        test_data = datasets.FashionMNIST('data', train=False, download=True,
                             transform= transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ]))
        
        if binary_loss:
            # Use only sandals and sneakers
            train_mask = (training_data.targets == 5) | (training_data.targets == 7)
            test_mask = (test_data.targets == 5) | (test_data.targets == 7)

            training_data.data = training_data.data[train_mask]
            training_data.targets = training_data.targets[train_mask]
            test_data.data = test_data.data[test_mask]
            test_data.targets = test_data.targets[test_mask]

            # Relabel classes `5` -> 0 and `7` -> 1
            training_data.targets = (training_data.targets == 7).long()
            test_data.targets = (test_data.targets == 7).long()

        # Create TensorDataset
        train_set = TensorDataset(training_data.data.type(torch.float).unsqueeze(1),
                                torch.tensor(training_data.targets, dtype=dtype_out))
        test_set = TensorDataset(test_data.data.type(torch.float).unsqueeze(1),
                                torch.tensor(test_data.targets, dtype=dtype_out))

        # Apply label noise to the training set for FashionMNIST
        if noise_rate > 0:
            train_set = apply_noise(train_set, noise_rate, seed_noise, dtype_out)

        # Print some data information
        labels = test_set.tensors[1]
        unique_classes, counts = torch.unique(labels, return_counts=True)
        total_samples = len(labels)
        proportions = counts / total_samples

        for cls, count, prop in zip(unique_classes, counts, proportions):
            print(f"Class {cls.item()}: Count = {count.item()}, Proportion = {prop.item():.2%}")

        return train_set, test_set
    
    elif name == "cifar10":
        
        from torchvision import datasets
        from torchvision import transforms
        
        training_data =   datasets.CIFAR10("data", train=True, download=True)

        test_data = datasets.CIFAR10('data', train=False, download=True)
        if binary_loss:
            # Select only instances belonging to classes `3` and `5` (cat and dogs)
            train_mask = (torch.tensor(training_data.targets) == 3) | (torch.tensor(training_data.targets) == 5)
            test_mask = (torch.tensor(test_data.targets) == 3) | (torch.tensor(test_data.targets) == 5)

            training_data.data = training_data.data[train_mask.numpy()]
            training_data.targets = torch.tensor(training_data.targets)[train_mask].numpy()
            test_data.data = test_data.data[test_mask.numpy()]
            test_data.targets = torch.tensor(test_data.targets)[test_mask].numpy()

            # Relabel classes `3` -> 0 and `5` -> 1
            training_data.targets = (np.array(training_data.targets) == 5).astype(np.uint8)
            test_data.targets = (np.array(test_data.targets) == 5).astype(np.uint8)

        # Normalize the data (and rearranges)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # CIFAR-10 normalization values
        ])

        train_data_tensor = torch.stack([transform(img) for img in training_data.data])
        test_data_tensor = torch.stack([transform(img) for img in test_data.data])

        # Create TensorDataset
        train_set = TensorDataset(train_data_tensor,
                                torch.tensor(training_data.targets, dtype=dtype_out))
        test_set = TensorDataset(test_data_tensor,
                                torch.tensor(test_data.targets, dtype=dtype_out))

        if noise_rate > 0:
            train_set = apply_noise(train_set, noise_rate, seed_noise, dtype_out)

        # Print some data information
        labels = torch.tensor(test_data.targets, dtype=dtype_out)
        unique_classes, counts = torch.unique(labels, return_counts=True)
        total_samples = len(labels)
        proportions = counts / total_samples

        for cls, count, prop in zip(unique_classes, counts, proportions):
            print(f"Class {cls.item()}: Count = {count.item()}, Proportion = {prop.item():.2%}")

        return train_set, test_set
    
        #train_set = TensorDataset(torch.Tensor(training_data.data).type(torch.float16).permute(0,3,1,2), torch.Tensor(training_data.targets).type(torch.uint8))
        #test_set = TensorDataset(torch.from_numpy(test_data.data).type(torch.float16).permute(0,3,1,2), torch.Tensor(test_data.targets).type(torch.uint8))

        #return train_set, test_set
    elif name == "adult":

        # Define file paths
        train_file = "data/adult/adult.data"
        test_file = "data/adult/adult.test"
     
        # Define column names based on `adult.names`
        columns = [
            "age", "workclass", "fnlwgt", "education", "education-num", 
            "marital-status", "occupation", "relationship", "race", 
            "sex", "capital-gain", "capital-loss", "hours-per-week", 
            "native-country", "income"
        ]       


        try:
            # Load training data
            train_data = pd.read_csv(train_file, header=None, names=columns, na_values=" ?", skipinitialspace=True)

            # Load test data (test file has an extra line and labels with a dot at the end)
            test_data = pd.read_csv(test_file, header=None, names=columns, na_values=" ?", skipinitialspace=True, skiprows=1)

            # Combine training and test data for preprocessing, if needed
            data = pd.concat([train_data, test_data], axis=0)
            
        except Exception:
            
            raise ValueError(f"Adult dataset is not inside the data folder! cwd {os.getcwd()}")
        
        # Drop rows with missing values
        data = data.dropna()
        # Clean up target variable (remove the trailing period in the test set)
        data['income'] = data['income'].replace({'<=50K.': '<=50K', '>50K.': '>50K'})
        # Convert the target ("income") to binary (0 for `<=50K`, 1 for `>50K`)
        data['income'] = data['income'].apply(lambda x: 1 if x == '>50K' else 0)

        # List of continuous and non-continuous features
        categorical_features = [
            'workclass', 'education', 'marital-status', 'occupation', 
            'relationship', 'race', 'sex', 'native-country'
        ]
        continuous_features = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
        target_feature = 'income'

        # Apply get_dummies to categorical features
        data_encoded = pd.get_dummies(data, columns=categorical_features, drop_first=True)

        # Reorder columns: first continuous features, then dummy variables, and finally the target
        ordered_columns = continuous_features + [col for col in data_encoded.columns if col != target_feature and col not in continuous_features] + [target_feature]

        # Reorder the DataFrame columns according to the desired order
        data = data_encoded[ordered_columns]

        # Display dataset information after preprocessing
        #print("Preprocessed Dataset Preview:")
        #print(data.columns)
        #print(data.head())

        print("\nDataset Shape:", data.shape)

        X_train, X_test, y_train, y_test = preprocess(data, preprocess_config, 'income')
        
        train_set = TensorDataset(torch.tensor(X_train, dtype=dtype_in), torch.tensor(y_train,dtype=dtype_out))
        test_set = TensorDataset(torch.tensor(X_test, dtype=dtype_in), torch.tensor(y_test, dtype=dtype_out))

        if noise_rate > 0:
            train_set = apply_noise(train_set, noise_rate, seed_noise, dtype_out)
        
        return train_set, test_set

    elif name =='ionosphere':
        from ucimlrepo import fetch_ucirepo 
    
        # fetch dataset 
        ionosphere = fetch_ucirepo(id=52) 
        
        # data (as pandas dataframes) 
        X = ionosphere.data.features 
        y = ionosphere.data.targets 
        # Convert 'g' to 1 and 'b' to 0
        y = y.replace({'g': 1, 'b': 0})
        df = pd.concat([X, y], axis=1)

        X_train, X_test, y_train, y_test = preprocess(df, preprocess_config, 'Class')

        train_set = TensorDataset(torch.tensor(X_train, dtype=dtype_in), torch.tensor(y_train,dtype=dtype_out))
        test_set = TensorDataset(torch.tensor(X_test, dtype=dtype_in), torch.tensor(y_test, dtype=dtype_out))

        return train_set, test_set

    elif name == 'phomene':
        try:
            df=pd.read_csv('data/phomene.csv')
            
        except Exception:
            
            raise ValueError(f"Phomene dataset is not inside the data folder! cwd {os.getcwd()}")
        df['Class'] = df['Class'].apply(lambda x: 0 if x == 1 else 1)

        X_train, X_test, y_train, y_test = preprocess(df, preprocess_config, 'Class')
        
        train_set = TensorDataset(torch.tensor(X_train, dtype=dtype_in), torch.tensor(y_train,dtype=dtype_out))
        test_set = TensorDataset(torch.tensor(X_test, dtype=dtype_in), torch.tensor(y_test, dtype=dtype_out))

        if noise_rate > 0:
            train_set = apply_noise(train_set, noise_rate, seed_noise, dtype_out)
        
        return train_set, test_set

    elif name == "creditcard":
        try:
            data, meta = arff.loadarff('data/creditcard.arff')
            df = pd.DataFrame(data)
        except Exception:
            raise ValueError(f"Creditcard dataset is not inside the data folder! cwd {os.getcwd()}")
        
        # Convert 'Class' from bytes to int (if necessary)
        df['Class'] = df['Class'].apply(lambda x: int(x.decode('utf-8')) if isinstance(x, bytes) else int(x))
        
        X_train, X_test, y_train, y_test = preprocess(df, preprocess_config, 'Class')
        
        train_set = TensorDataset(torch.tensor(X_train, dtype=dtype_in), torch.tensor(y_train, dtype=dtype_out))
        test_set = TensorDataset(torch.tensor(X_test, dtype=dtype_in), torch.tensor(y_test, dtype=dtype_out))
        
        return train_set, test_set

    else:
        
        raise ValueError(f"Dataset {name} is not available!")
            
    
    
if __name__ == "__main__":
    from ucimlrepo import fetch_ucirepo 
    
    # fetch dataset 
    ionosphere = fetch_ucirepo(id=52) 
    
    # data (as pandas dataframes) 
    X = ionosphere.data.features 
    y = ionosphere.data.targets 
    # Convert 'g' to 1 and 'b' to 0
    y = y.replace({'g': 1, 'b': 0})
    df = pd.concat([X, y], axis=1)

    dummy_param_bin = True
    dummy_param_param = {'resample': 1, 'poly_features_degree': 2, 'scaler': 'Standard', "seed_split":42}
    dtype_in = torch.float32
    dtype_out = torch.float32 if dummy_param_bin else torch.long


    X_train, X_test, y_train, y_test = preprocess(df, dummy_param_param, 'Class')
    print(y_test[y_test == 1].sum() / len(y_test))
    print(type(y_train), y_train.shape, y_train[0:5])
    print(dtype_out)

    train_set = TensorDataset(torch.tensor(X_train, dtype=dtype_in), torch.tensor(y_train,dtype=dtype_out))
    test_set = TensorDataset(torch.tensor(X_test, dtype=dtype_in), torch.tensor(y_test, dtype=dtype_out))
