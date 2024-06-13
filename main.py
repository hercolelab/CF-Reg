import multiprocessing
import torch
from torch.utils.data import DataLoader
import numpy as np
from typing import List, Tuple
from src.estimator import MontecarloEstimator
from src.trainer import LightningClassifier
from src.utility import get_dataset, get_model, get_loss, merge_hydra_wandb, ClassifierEvaluator, read_yaml
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities import disable_possible_user_warnings
from src.utility.utils import flatten_dict

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

    run_name: str = f"{cfg.model.model_type}_{cfg.data.name}_{cfg.loss.type}_{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    run.name = run_name
    run.save()

def is_counterfactual(cfg):
    
    return True if cfg.loss.type != "normal" else False

def run_sweep_agent(cfg: DictConfig, sweep_id: str):
    wandb.agent(sweep_id=sweep_id, function=lambda: train(cfg))
    
    
def train(cfg):
        
        with wandb.init(project=cfg.logger.project, mode=cfg.logger.mode)  as run: 

            merge_hydra_wandb(cfg, wandb.config)
            log_params(cfg)
            set_run_name(cfg, run)
            
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

            trainset, testset = get_dataset(name=cfg.data.name) 
            model = get_model(config=OmegaConf.to_container(cfg.model) | {"input_dim": cfg.data.input_dim, "nclasses": cfg.data.nclasses, "channel_in": cfg.data.channel_in})
            criterion = get_loss(**cfg.loss)
            estimator = MontecarloEstimator(function=model, train_set=trainset, **cfg.estimator)
            evaluator = ClassifierEvaluator(classes=cfg.data.nclasses)
            
            clf =  LightningClassifier(model=model, 
                                       criterion=criterion, 
                                       optim_config=OmegaConf.to_container(cfg.optimizer), 
                                       evaluator=evaluator, 
                                       estimator=estimator, 
                                       counterfactual=is_counterfactual(cfg))
                
            train_loader = DataLoader(trainset, **cfg.loader)
            test_loader = DataLoader(testset, **cfg.loader)
            
            wandb_logger.watch(model, log='gradients', log_freq=100)
            trainer = pl.Trainer(**cfg.trainer, logger=wandb_logger)
            trainer.fit(clf, train_loader, test_loader)

@hydra.main(version_base="1.3", config_path="hydra_configs", config_name="config")
def main(cfg: DictConfig) -> None:
    
    
    if cfg.run_mode == 'sweep':

        sweep_config = read_yaml(f'wandb_sweeps_configs/{cfg.logger.config}.yaml')
        sweep_id = wandb.sweep(sweep=sweep_config, project=cfg.logger.project)
        
        multiprocessing.set_start_method('spawn', force=True)

        # Number of parallel agents
        num_agents = cfg.num_agents if 'num_agents' in cfg else 4 # Default to 4 agents if not specified
        processes = []
        for _ in range(num_agents):
            p = multiprocessing.Process(target=run_sweep_agent, args=(cfg, sweep_id))
            p.start()
            processes.append(p)
        
        for p in processes:
            p.join()
        
    elif cfg.run_mode == "run":
        
        train()
        
    else:
        
        raise ValueError(f"Values for run_mode can be sweep or run, you insert {cfg.run_mode}")

if __name__ == "__main__":
    
    main()


    

#TODO: aggiungi regolarizzazione l1 e dropout
#TODO: aggiungi la funzione simil-relu alle immagini in greyscale
