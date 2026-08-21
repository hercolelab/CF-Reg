import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
import numpy as np
import random
from typing import Any, Mapping
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


def merge_train_validation_sets(
    train_set: TensorDataset,
    validation_set: TensorDataset,
) -> TensorDataset:
    """Return the final-training set without including any holdout samples."""

    if len(train_set.tensors) != len(validation_set.tensors):
        raise ValueError("Training and validation sets must contain the same number of tensors")

    return TensorDataset(
        *(
            torch.cat((train_tensor, validation_tensor), dim=0)
            for train_tensor, validation_tensor in zip(
                train_set.tensors,
                validation_set.tensors,
            )
        )
    )


def select_datasets_for_mode(
    train_set: TensorDataset,
    validation_set: TensorDataset,
    test_set: Dataset,
    tuning: bool,
    early_stopping_enabled: bool,
) -> tuple[TensorDataset, Dataset | None]:
    """Select exactly the datasets allowed by the requested execution mode."""

    if tuning or early_stopping_enabled:
        return train_set, None if tuning else test_set

    return merge_train_validation_sets(train_set, validation_set), test_set


def fit_and_evaluate(
    trainer: Any,
    classifier: LightningClassifier,
    fit_set: Dataset,
    validation_set: Dataset,
    test_set: Dataset | None,
    loader_config: Mapping[str, Any],
    tuning: bool,
    early_stopping_enabled: bool,
) -> None:
    """Run either tuning (train/validation) or final fit/test evaluation."""

    if not tuning and test_set is None:
        raise ValueError("A holdout test set is required when tuning is disabled")

    use_validation = tuning or early_stopping_enabled
    train_loader = DataLoader(fit_set, **dict(loader_config))
    evaluation_loader_config = dict(loader_config)
    evaluation_loader_config["shuffle"] = False

    if use_validation:
        validation_loader = DataLoader(validation_set, **evaluation_loader_config)
        trainer.fit(classifier, train_loader, validation_loader)
    else:
        trainer.fit(classifier, train_loader)

    if tuning:
        return

    test_loader = DataLoader(test_set, **evaluation_loader_config)
    checkpoint = "best" if early_stopping_enabled else None
    trainer.test(classifier, dataloaders=test_loader, ckpt_path=checkpoint)


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
    #run.save()

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
            # To increase performances on CUDA 
            torch.set_float32_matmul_precision('high')
            wandb_logger = WandbLogger(project=cfg.logger.project)
            
            random.seed(cfg.seed)
            np.random.seed(cfg.seed) 
            torch.manual_seed(cfg.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(cfg.seed)
                torch.cuda.manual_seed_all(cfg.seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            print(cfg)
            print("cfg.seed: ", cfg.seed)
            
            callback_config = OmegaConf.to_container(cfg.trainer.callbacks) or {}
            early_stopping_enabled = bool(
                callback_config.get("early_stop_enable", False)
            )

            trainset, validationset, testset = get_dataset(
                name=cfg.data.name,
                binary=cfg.loss.binary,
                preprocess_config=OmegaConf.to_container(cfg.preprocessor),
                seed=cfg.seed,
            )

            fit_set, holdout_testset = select_datasets_for_mode(
                train_set=trainset,
                validation_set=validationset,
                test_set=testset,
                tuning=cfg.tuning,
                early_stopping_enabled=early_stopping_enabled,
            )

            # TODO These preprocessing steps should ideally be refactored into get_dataset().
            # Extract the tensors from the training set
            train_data, train_targets = trainset[:][0], trainset[:][1]
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
            estimator = get_estimator(**(OmegaConf.to_container(cfg.estimator) | {"function" : model, "train_set" : fit_set}))
            print("type(estimator): ", type(estimator))

            criterion = get_loss(**(OmegaConf.to_container(cfg.loss) | {"function" : model, "train_set" : fit_set}))
            evaluator = ClassifierEvaluator(classes=cfg.data.nclasses)
            
            clf =  LightningClassifier(model=model, 
                                       criterion=criterion, 
                                       optim_config=OmegaConf.to_container(cfg.optimizer), 
                                       evaluator=evaluator, 
                                       estimator=estimator, 
                                       counterfactual=is_counterfactual(cfg),
                                        margin = False)
                
            wandb_logger.watch(model, log='gradients', log_freq=100)
 

            
            trainer_cfg = {k: v for k, v in cfg.trainer.items() if k != "callbacks"}

            callbacks = get_callbacks(**callback_config)
            trainer = pl.Trainer(**trainer_cfg, callbacks=callbacks, logger=wandb_logger)
            fit_and_evaluate(
                trainer=trainer,
                classifier=clf,
                fit_set=fit_set,
                validation_set=validationset,
                test_set=holdout_testset,
                loader_config=OmegaConf.to_container(cfg.loader),
                tuning=cfg.tuning,
                early_stopping_enabled=early_stopping_enabled,
            )
    
    
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
