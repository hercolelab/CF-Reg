from typing import List
import torch
from src.estimator import Estimator
import pytorch_lightning as L
from src.models.models import extract_embeddings_hook
from src.utility.evaluation import ClassifierEvaluator
from src.utility.optimizer import get_optimizer
import numpy as np
import inspect
import time

class LightningClassifier(L.LightningModule):
    
    def __init__(self, 
                 model: torch.nn.Module, 
                 criterion: torch.nn.Module,
                 optim_config: dict,
                 evaluator: ClassifierEvaluator,
                 estimator: Estimator,
                 counterfactual: bool,
                 margin: bool) -> None:
        
        super().__init__()

        self.model = model
        self.optim_config = optim_config
        self.criterion = criterion
        self.train_output = []
        self.train_target = []
        self.train_loss = []
        self.train_p_x = []
        self.val_output = []
        self.val_target = []
        self.val_loss = []
        self.val_p_x = []
        self.test_output = []
        self.test_target = []
        self.test_loss = []
        self.test_samples = 0
        self.test_p_x = []
        self.train_embeddings = []
        self.test_embeddings = []
        self.evaluator = evaluator
        self.estimator = estimator
        self.counterfactual = counterfactual
        self.show_embedding = False
        self.margin = margin
        self.all_train_margins: List[List[float]] = []
        
        if self.show_embedding:
            self.model.layers[-2].register_forward_hook(extract_embeddings_hook)


#    def backward(self, loss: torch.Tensor, optimizer: torch.optim.Optimizer, optimizer_idx: int, *args, **kwargs):
#        super().backward(loss, optimizer, optimizer_idx, *args, **kwargs)
#        for name, param in self.model.named_parameters():
#            if param.grad is not None:
#                print(f"{name}: {param.grad.norm()}")  # Print gradient norm
#            else:
#                print(f"{name}: No gradient found")    
    
    def configure_optimizers(self):
        
        return get_optimizer(params=self.model.parameters(), config=self.optim_config)
        
        
    def on_train_epoch_start(self) -> None:
        self.train_t_start = time.time_ns()
        self.train_output = []
        self.train_target = []
        self.train_loss = []
        self.train_estimate = []
        self.train_margin = []
    
    def on_train_epoch_end(self) -> None:
        self.train_t_end = time.time_ns()
        stage: str = "train"
        with torch.no_grad():
            accuracy, f1, precision, recall, crossentropy = self.evaluator.get_complete_evaluation(self.train_output, self.train_target)
        
        if self.margin:
            evcp_bound = self.evaluator.get_avg_evcp_bound(np.mean(self.train_margin), self.estimator.radius, 5005) if np.mean(self.train_margin) <= self.estimator.radius else 0
   
    
        log_data = {
            f"{stage}/loss": sum(self.train_loss) / len(self.train_loss),
            f"{stage}/epoch": self.current_epoch,
            f"{stage}/accuracy": accuracy,
            f"{stage}/f1-score": f1,
            f"{stage}/precision": precision,
            f"{stage}/recall": recall,
            f"{stage}/crossentropy": crossentropy,
            f"{stage}/time_elapsed" : self.train_t_end - self.train_t_start
        }

        estimator_log_data = self.estimator.build_log(self.train_estimate, stage)

        if self.margin:
            log_data.update({f"{stage}/avgmargin" : np.mean(self.train_margin),
            f"{stage}/avgevcpbound": evcp_bound})
            # **NEW**: store a copy of this epoch’s estimates
            self.all_train_margins.append(self.train_margin.copy())

        log_data.update(estimator_log_data)

        self.log_dict(log_data, on_epoch=True, on_step=False)

 
        
        
    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:

        data, target = batch
        output = self.model(data)
        
        #out, target_cf = self.estimator.get_counterfactual(data, output, grad=self.counterfactual)
        #p_x = self.estimator.counterfactual_probability(out=out, target=target_cf)
        #if self.counterfactual:
        #    values = values | { "out_cf": out, "target_cf": target_cf}

        estimate = self.estimator.get_estimate(data = data, output = output)
        values: dict = {"input": output, "target": target, "estimate": estimate, "weights": self.model.parameters(), "data": data}

        #forward_signature = list(inspect.signature(self.criterion.__class__.forward).parameters.keys())[1:] # the first parameter is self, so it can be dropped
        #values = {key: value for key,value in values.items() if key in forward_signature}


        torch.set_grad_enabled(mode=True)
        loss = self.criterion(**values)       
        #print("batch_idx:", batch_idx)
        #for name, param in self.model.named_parameters():
        #    if param.grad is not None:
        #        print(f"{name}: {param.grad.norm()}")
        #    else:
        #        print(f"{name}: No gradient found")
      
        self.train_target += target.tolist()
        self.train_output += output.tolist()
        self.train_loss += [loss.item()]
        self.train_estimate += estimate.tolist()

        if self.margin:
            torch.set_grad_enabled(mode=False)
            w = self.model.linear.weight.cpu()
            f_x = self.model.forward(data).cpu()
            #print("f_x.shape:", f_x.shape)
            #print("w.shape:", w.shape)
            margin = np.abs(f_x/np.linalg.norm(w))
            self.train_margin += margin.tolist()
            #print("margin:", margin.shape)
            torch.set_grad_enabled(mode=True)

        if self.show_embedding:
            from sklearn.decomposition import PCA
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            
            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_subplot(111, projection='3d')
            pca: PCA = PCA(n_components=3)
            pca_components = pca.fit_transform(self.model.layers[-2].embeddings.detach().cpu().numpy())
            ax.scatter(pca_components[:100, 0], pca_components[:100, 1], pca_components[:100, 2], c=torch.argmax(out[:100], dim=1).detach().cpu().numpy())
            plt.savefig(f"imgs/pca_{self.current_epoch}_{batch_idx}.png")
            plt.close()
        return loss
    
    def on_train_end(self) -> None:
        # Dump the full list-of-lists to a pickle
        if self.margin:
            import pickle
            filename = "all_train_margins" + str(self.estimator.radius) + ".pkl"
            with open(filename , "wb") as f:
                pickle.dump(self.all_train_margins, f)
    
    def on_validation_epoch_start(self) -> None:
        self.val_t_start = time.time_ns()
        self.val_output = []
        self.val_target = []
        self.val_loss = []
        self.val_estimate = []
    
    
    def on_validation_epoch_end(self) -> None:
        self.val_t_end = time.time_ns()
        if self.trainer.state.stage != "sanity_check":
            
            stage: str = "validation"
            accuracy, f1, precision, recall, crossentropy = self.evaluator.get_complete_evaluation(self.val_output, self.val_target)
            
            log_data = {
                f"{stage}/loss": sum(self.val_loss) / len(self.val_loss),
                f"{stage}/epoch": self.current_epoch,
                f"{stage}/accuracy": accuracy,
                f"{stage}/f1-score": f1,
                f"{stage}/precision": precision,
                f"{stage}/recall": recall,
                f"{stage}/crossentropy": crossentropy,
                f"{stage}/time_elapsed" : self.val_t_end - self.val_t_start
            }

            estimator_log_data = self.estimator.build_log(self.val_estimate, stage)

            log_data.update(estimator_log_data)

            self.log_dict(log_data, on_epoch=True, on_step=False) 



    def validation_step(self, batch, batch_idx):
        
  
        data, target = batch
        output = self.model(data)
        #values: dict = {"input": output, "target": target}
        #out, target_cf = self.estimator.get_counterfactual(data, output, grad=False)
        #p_x = self.estimator.get_estimate(out=out, target=target_cf)
   
#        old_params = {name: param.clone() for name, param in self.model.named_parameters()}
        with torch.enable_grad():
            estimate = self.estimator.get_estimate(data = data, output = output)
        #  new_params = {name: param for name, param in self.model.named_parameters()}

        values: dict = {"input": output, "target": target, "estimate": estimate, "weights": self.model.parameters(), "data": data}
#        forward_signature = list(inspect.signature(self.criterion.__class__.forward).parameters.keys())[1:] # the first parameter is self, so it can be dropped
#        values = {key: value for key,value in values.items() if key in forward_signature}
        #if self.counterfactual:
        #    values = values | { "out_cf": out, "target_cf": target_cf}        
     
        val_loss = self.criterion(**values)   
        self.val_target += target.tolist()
        self.val_output += output.tolist()
        self.val_loss += [val_loss.item()]   
        self.val_estimate += estimate.tolist()

#        for name in old_params:
#            if not torch.equal(old_params[name], new_params[name].data):
#                print(f"Parameter '{name}' has changed.")
#            else:
#                print(f"Parameter '{name}' is unchanged.")
        return val_loss

    def on_test_epoch_start(self) -> None:
        self.test_t_start = time.time_ns()
        self.test_output = []
        self.test_target = []
        self.test_loss = []
        self.test_samples = 0
        self.test_estimate = []

    def on_test_epoch_end(self) -> None:
        self.test_t_end = time.time_ns()
        stage: str = "test"
        accuracy, f1, precision, recall, crossentropy = self.evaluator.get_complete_evaluation(self.test_output, self.test_target)

        log_data = {
            f"{stage}/loss": sum(self.test_loss) / self.test_samples,
            f"{stage}/epoch": self.current_epoch,
            f"{stage}/accuracy": accuracy,
            f"{stage}/f1-score": f1,
            f"{stage}/precision": precision,
            f"{stage}/recall": recall,
            f"{stage}/crossentropy": crossentropy,
            f"{stage}/time_elapsed" : self.test_t_end - self.test_t_start
        }

        estimator_log_data = self.estimator.build_log(self.test_estimate, stage)
        log_data.update(estimator_log_data)

        self.log_dict(log_data, on_epoch=True, on_step=False)

    def test_step(self, batch, batch_idx):
        data, target = batch
        output = self.model(data)

        with torch.enable_grad():
            estimate = self.estimator.get_estimate(data=data, output=output)

        values: dict = {"input": output, "target": target, "estimate": estimate, "weights": self.model.parameters(), "data": data}
        test_loss = self.criterion(**values)

        self.test_target += target.tolist()
        self.test_output += output.tolist()
        self.test_loss += [test_loss.item() * target.size(0)]
        self.test_samples += target.size(0)
        self.test_estimate += estimate.tolist()

        return test_loss
