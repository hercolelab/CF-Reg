import torch
from src.estimator.montecarlo import MontecarloEstimator
import pytorch_lightning as L
from src.models.models import extract_embeddings_hook
from src.utility.evaluation import ClassifierEvaluator
from src.utility.optimizer import get_optimizer



class LightningClassifier(L.LightningModule):
    
    def __init__(self, 
                 model: torch.nn.Module, 
                 criterion: torch.nn.Module,
                 optim_config: dict,
                 evaluator: ClassifierEvaluator,
                 estimator: MontecarloEstimator,
                 counterfactual: bool) -> None:
        
        super().__init__()

        self.model = model
        self.pnoise = optim_config.pop("pnoise")
        self.optim_config = optim_config
        self.criterion = criterion
        self.train_output = []
        self.train_target = []
        self.train_loss = []
        self.train_p_x = []
        self.train_e_z = []
        self.val_output = []
        self.val_target = []
        self.val_loss = []
        self.val_p_x = []
        self.val_e_z = []
        self.train_embeddings = []
        self.test_embeddings = []
        self.evaluator = evaluator
        self.estimator = estimator
        self.counterfactual = counterfactual
        self.show_embedding = False
        
        if self.show_embedding:
            self.model.layers[-2].register_forward_hook(extract_embeddings_hook)

        
    def configure_optimizers(self):
        
        return get_optimizer(params=self.model.parameters(), config=self.optim_config)
        
        
    def on_train_epoch_start(self) -> None:
        
        self.train_output = []
        self.train_target = []
        self.train_loss = []
        self.train_p_x = []
        self.train_e_z = []
    
    def on_train_epoch_end(self) -> None:
        
        stage: str = "train"
        accuracy, f1, precision, recall = self.evaluator.get_complete_evaluation(self.train_output, self.train_target)
        
        self.log_dict({f"{stage}/loss": sum(self.train_loss)/len(self.train_loss), 
                       f"{stage}/epoch": self.current_epoch, 
                       f"{stage}/accuracy": accuracy, 
                       f"{stage}/f1-score": f1, 
                       f"{stage}/precision": precision, 
                       f"{stage}/recall": recall,
                       f"{stage}/E_z": sum(self.train_e_z),
                       f"{stage}/p_x": sum(self.train_p_x)/len(self.train_p_x)}, on_epoch=True, on_step=False)  
        
        
    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
                
        use_noise = torch.rand(1).item() < self.pnoise
        
        data, target = batch
        
        if use_noise:
            
            target = torch.cat((target.unsqueeze(1).repeat(1, self.estimator.n_samples).view(-1), target))

        # TODO: controlla come viene calcolata l'accuracy
            
        output = self.model(data, use_noise)
        values: dict = {"input": output, "target": target}
        #out, target_cf = self.estimator.get_counterfactual(data=data, target=output, grad=self.counterfactual)
        out, target_cf = self.estimator.get_counterfactual(data=data, target=self.model(data, False), grad=False)
        p_x, e_z = self.estimator.counterfactual_probability(out=out, target=target_cf)
        if self.counterfactual:
            values = values | { "out_cf": out, "target_cf": target_cf}
            
        torch.set_grad_enabled(mode=True)
        loss = self.criterion(**values)        
        self.train_target += target[-data.shape[0]:].tolist()
        self.train_output += output[-data.shape[0]:].tolist()
        self.train_loss += [loss.item()]
        self.train_p_x += p_x.tolist()
        self.train_e_z += [e_z.item()]
                
        return loss
    
    def on_validation_epoch_start(self) -> None:
        
        self.val_output = []
        self.val_target = []
        self.val_loss = []
        self.val_p_x = []
        self.val_e_z = []
    
    def on_validation_epoch_end(self) -> None:
        
        if self.trainer.state.stage != "sanity_check":
            
            stage: str = "validation"
            accuracy, f1, precision, recall = self.evaluator.get_complete_evaluation(self.val_output, self.val_target)
            
            self.log_dict({f"{stage}/loss": sum(self.val_loss)/len(self.val_loss), 
                        f"{stage}/epoch": self.current_epoch, 
                        f"{stage}/accuracy": accuracy, 
                        f"{stage}/f1-score": f1, 
                        f"{stage}/precision": precision, 
                        f"{stage}/recall": recall,
                        f"{stage}/E_z": sum(self.val_e_z),
                        f"{stage}/p_x": sum(self.val_p_x)/len(self.val_p_x)}, on_epoch=True, on_step=False) 

    def validation_step(self, batch, batch_idx):
        
        use_noise = torch.rand(1).item() < self.pnoise
        
        data, target = batch
        
        if use_noise:
            
            target = torch.cat((target.unsqueeze(1).repeat(1, self.estimator.n_samples).view(-1), target))

        output = self.model(data, use_noise)
        values: dict = {"input": output, "target": target}
        #out, target_cf = self.estimator.get_counterfactual(data, output, grad=False)
        out, target_cf = self.estimator.get_counterfactual(data=data, target=self.model(data, False), grad=False)
        p_x, e_z = self.estimator.counterfactual_probability(out=out, target=target_cf)
        if self.counterfactual:
            values = values | { "out_cf": out, "target_cf": target_cf}        

        val_loss = self.criterion(**values)   
        self.val_target += target[-data.shape[0]:].tolist()
        self.val_output += output[-data.shape[0]:].tolist()
        self.val_loss += [val_loss.item()]   
        self.val_p_x += p_x.tolist()
        self.val_e_z += [e_z.item()]
        return val_loss
