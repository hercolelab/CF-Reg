import torch
from src.estimator import Estimator
import pytorch_lightning as L
from src.models.models import extract_embeddings_hook
from src.utility.evaluation import ClassifierEvaluator
from src.utility.optimizer import get_optimizer
import numpy as np
import inspect
import time
from typing import Tuple

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
        self.train_embeddings = []
        self.test_embeddings = []
        self.evaluator = evaluator
        self.estimator = estimator
        self.counterfactual = counterfactual
        self.show_embedding = False
        self.margin = margin
        
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
        self.train_original_target = [] # Reset for each epoch
        self.train_is_noisy_label = [] # Reset for each epoch
    
    def on_train_epoch_end(self) -> None:
        self.train_t_end = time.time_ns()
        stage: str = "train"
        with torch.no_grad():
            # Overall accuracy
            overall_accuracy, overall_f1, overall_precision, overall_recall, overall_crossentropy = \
                self.evaluator.get_complete_evaluation(self.train_output, self.train_target)
            
            log_data = {
                f"{stage}/loss": sum(self.train_loss) / len(self.train_loss),
                f"{stage}/epoch": self.current_epoch,
                f"{stage}/overall_accuracy": overall_accuracy, # Renamed for clarity
                f"{stage}/overall_f1-score": overall_f1,
                f"{stage}/overall_precision": overall_precision,
                f"{stage}/overall_recall": overall_recall,
                f"{stage}/overall_crossentropy": overall_crossentropy,
                f"{stage}/time_elapsed" : self.train_t_end - self.train_t_start
            }

            # Calculate accuracy on clean and noisy data if available
            if self.train_original_target and self.train_is_noisy_label:
                original_targets = torch.tensor(self.train_original_target)
                current_targets = torch.tensor(self.train_target)
                outputs = torch.tensor(self.train_output)
                is_noisy = torch.tensor(self.train_is_noisy_label, dtype=torch.bool)

                # Filter clean data
                clean_outputs = outputs[~is_noisy]
                clean_original_targets = original_targets[~is_noisy]
                
                # Filter noisy data (where the original label was different from the current label)
                # This specifically targets labels that were intentionally flipped
                actually_flipped_mask = (original_targets != current_targets) & is_noisy
                noisy_outputs = outputs[actually_flipped_mask]  # predictions for noisy data
                noisy_original_targets = original_targets[actually_flipped_mask] # Use original targets for 'noisy' data to evaluate against true labels

                if len(clean_outputs) > 0:
                    clean_accuracy, _, _, _, _ = self.evaluator.get_complete_evaluation(clean_outputs.tolist(), clean_original_targets.tolist())
                    log_data[f"{stage}/clean_accuracy"] = clean_accuracy
                else:
                    log_data[f"{stage}/clean_accuracy"] = torch.nan # No clean data points

                if len(noisy_outputs) > 0:
                    # For noisy data, we want to see how well the model predicts the *original* (correct) label
                    noisy_accuracy, _, _, _, _ = self.evaluator.get_complete_evaluation(noisy_outputs.tolist(), noisy_original_targets.tolist())
                    log_data[f"{stage}/noisy_accuracy_vs_original_label"] = noisy_accuracy
                    
                    # You might also be interested in how well it predicts the *flipped* label
                    noisy_flipped_targets = current_targets[actually_flipped_mask]
                    noisy_accuracy_vs_flipped, _, _, _, _ = self.evaluator.get_complete_evaluation(noisy_outputs.tolist(), noisy_flipped_targets.tolist())
                    log_data[f"{stage}/noisy_accuracy_vs_flipped_label"] = noisy_accuracy_vs_flipped
                else:
                    log_data[f"{stage}/noisy_accuracy_vs_original_label"] = torch.nan
                    log_data[f"{stage}/noisy_accuracy_vs_flipped_label"] = torch.nan


            if self.margin:
                evcp_bound = self.evaluator.get_avg_evcp_bound(np.mean(self.train_margin), self.estimator.radius, 5005) if np.mean(self.train_margin) <= self.estimator.radius else 0
       
            estimator_log_data = self.estimator.build_log(self.train_estimate, stage)

            if self.margin:
                log_data.update({f"{stage}/avgmargin" : np.mean(self.train_margin),
                f"{stage}/avgevcpbound": evcp_bound})

            log_data.update(estimator_log_data)

            self.log_dict(log_data, on_epoch=True, on_step=False)  
        
        
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        # Unpack batch: data, current_target, original_target, is_noisy_label
        # original_target and is_noisy_label will be None if noise_rate is 0
        if len(batch) == 4:
            data, target, original_target, is_noisy_label = batch
        else: # Handle cases where noise isn't applied (original code behavior)
            data, target = batch
            original_target = None
            is_noisy_label = None

        output = self.model(data)
        
        #out, target_cf = self.estimator.get_counterfactual(data, output, grad=self.counterfactual)
        #p_x = self.estimator.counterfactual_probability(out=out, target=target_cf)
        #if self.counterfactual:
        #    values = values | { "out_cf": out, "target_cf": target_cf}

        estimate = self.estimator.get_estimate(data = data, output = output)
        #estimate = None
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

        if original_target is not None:
            self.train_original_target += original_target.tolist()
            self.train_is_noisy_label += is_noisy_label.tolist()

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
        torch.set_grad_enabled(mode=True)
        estimate = self.estimator.get_estimate(data = data, output = output)
        #estimate = None
        torch.set_grad_enabled(mode=False)
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
