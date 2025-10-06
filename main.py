import torch
from torch.utils.data import DataLoader
import numpy as np
from typing import List, Tuple
from src.estimator import MontecarloEstimator, SCFEEstimator
from src.trainer import LightningClassifier
from src.utility import get_dataset, get_model, get_loss, get_estimator, merge_hydra_wandb, ClassifierEvaluator, read_yaml, get_callbacks
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities import disable_possible_user_warnings
from src.utility.utils import flatten_dict
from sklearn.preprocessing import PolynomialFeatures

disable_possible_user_warnings()

def log_params(cfg: DictConfig) -> None:
    
    import pandas as pd
    
        
    temp_config = OmegaConf.to_container(cfg)
    config_to_log = flatten_dict(d=temp_config)
    config_to_log = pd.DataFrame([config_to_log])
    config_to_log.astype(str)
    param_table = wandb.Table(dataframe=config_to_log)

    wandb.log({"params": param_table})
    
def set_run_name(cfg, run):
    
    from datetime import datetime
    
    if cfg.loss.type == "normal":
        loss_name = "CE"
    elif cfg.loss.type == "scfe_regularization":
        loss_name = "SCFE"
    else:   
        loss_name = cfg.loss.type

    if cfg.model.model_type == "BPreActResNet":
        model_name = "ResNet"
    else:
        model_name = cfg.model.model_type

    run_name: str = f"{model_name}_{cfg.data.name}_{loss_name}_{cfg.preprocessor.noise_rate}_{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    run.name = run_name
    run.save()

def is_counterfactual(cfg):
    
    return True if cfg.loss.type != "normal" else False


@hydra.main(version_base="1.3", config_path="hydra_configs", config_name="config")
def main(cfg: DictConfig) -> None:
    
    def train():
        
        with wandb.init(project=cfg.logger.project, mode=cfg.logger.mode)  as run: 
            #print("cfg: ", cfg)
            #print("wandb.config: ", wandb.config)
            merge_hydra_wandb(cfg, wandb.config)
            log_params(cfg)
            set_run_name(cfg, run)
            #cfg.preprocessor['seed_split'] = cfg.seed
            # To increase performances on CUDA 
            torch.set_float32_matmul_precision('high')
            wandb_logger = WandbLogger(project=cfg.logger.project)
            
            np.random.seed(cfg.seed) 
            torch.manual_seed(cfg.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(cfg.seed)
                torch.cuda.manual_seed_all(cfg.seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            print(cfg)
            print("cfg.seed: ", cfg.seed)
            
            trainset, testset = get_dataset(name = cfg.data.name, binary = cfg.loss.binary, preprocess_config = OmegaConf.to_container(cfg.preprocessor)) 

            # TODO These preprocessing steps should ideally be refactored into get_dataset().
            from torch.utils.data import TensorDataset
            # Extract the tensors from the trainset and testset
            train_data, train_targets = trainset[:][0], trainset[:][1]
            test_data, test_targets = testset[:][0], testset[:][1]
            print("train_data.shape: ", train_data.shape)
            # Apply PolynomialFeatures to the dataset
            #poly = PolynomialFeatures(degree=cfg.data.poly_degree)

            # Transform the data using PolynomialFeatures
            #train_data_poly = torch.tensor(poly.fit_transform(train_data.numpy()), dtype=torch.float32)
            #test_data_poly = torch.tensor(poly.transform(test_data.numpy()), dtype=torch.float32)

            # Create new TensorDatasets with the transformed data
            #trainset = TensorDataset(train_data_poly, train_targets)
            #testset = TensorDataset(test_data_poly, test_targets)

            # Update the input dimension in the model
           # cfg.data.input_dim = train_data_poly.size(1)


            model = get_model(config=OmegaConf.to_container(cfg.model) | {"input_dim": train_data.shape, "nclasses": cfg.data.nclasses, "channel_in": cfg.data.channel_in})
            estimator = get_estimator(**(OmegaConf.to_container(cfg.estimator) | {"function" : model, "train_set" : trainset}))
            print("type(estimator): ", type(estimator))

            criterion = get_loss(**(OmegaConf.to_container(cfg.loss) | {"function" : model, "train_set" : trainset}))
            evaluator = ClassifierEvaluator(classes=cfg.data.nclasses)
            
            clf =  LightningClassifier(model=model, 
                                       criterion=criterion, 
                                       optim_config=OmegaConf.to_container(cfg.optimizer), 
                                       evaluator=evaluator, 
                                       estimator=estimator, 
                                       counterfactual=is_counterfactual(cfg),
                                        margin = False)
                
            train_loader = DataLoader(trainset, **cfg.loader)
            test_loader = DataLoader(testset, **cfg.loader)
            
            wandb_logger.watch(model, log='gradients', log_freq=100)
 

            
            callbacks = cfg.trainer.callbacks
            trainer_cfg = {k: v for k, v in cfg.trainer.items() if k != "callbacks"}

            callbacks = get_callbacks(**callbacks)
            trainer = pl.Trainer(**trainer_cfg, callbacks=callbacks, logger=wandb_logger)
            trainer.fit(clf, train_loader, test_loader)
    
    
    if cfg.run_mode == 'sweep':
        print("Proect: ", cfg.logger.project)
        sweep_config = read_yaml(f'wandb_sweeps_configs/{cfg.logger.config}.yaml')
        sweep_id = wandb.sweep(sweep=sweep_config, project=cfg.logger.project)
        wandb.agent(sweep_id=sweep_id, function=train)
        
    elif cfg.run_mode == "run":
        
        train()
        
    else:
        
        raise ValueError(f"Values for run_mode can be sweep or run, you insert {cfg.run_mode}")

if __name__ == "__main__":
    #print(torch.__version__)
    #print(torch.cuda.is_available())
    main()


    

#TODO: aggiungi regolarizzazione l1 e dropout
#TODO: aggiungi la funzione simil-relu alle immagini in greyscale
