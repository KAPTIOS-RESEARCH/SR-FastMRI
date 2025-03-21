import wandb, torch, os, logging
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingLR, ReduceLROnPlateau
from src.utils.config import instanciate_module
from src.optimisation.early_stopping import EarlyStopping

class BaseTrainer(object):

    def __init__(self, model: nn.Module, parameters: dict, device: str):
        self.model = model
        self.parameters = parameters
        self.device = device
        self.early_stop = EarlyStopping(patience=parameters['early_stopping_patience'], enable_wandb=parameters['track'])
        
        # OPTIMIZER
        self.optimizer = Adam(
            self.model.parameters(), 
            lr=parameters['lr'],
            weight_decay=parameters['weight_decay']
        )
        
        # LR SCHEDULER
        self.lr_scheduler = None
        lr_scheduler_type = parameters['lr_scheduler'] if 'lr_scheduler' in parameters.keys() else 'none'

        if lr_scheduler_type == 'cosine':
            self.lr_scheduler = CosineAnnealingLR(optimizer=self.optimizer, T_max=100)
        elif lr_scheduler_type == 'plateau':
            self.lr_scheduler = ReduceLROnPlateau(optimizer=self.optimizer, mode='min', factor=0.1)
        elif lr_scheduler_type == 'exponential':
            self.lr_scheduler = ExponentialLR(optimizer=self.optimizer, gamma=0.97)

        # LOSS FUNCTION
        self.criterion = instanciate_module(parameters['loss']['module_name'],
                                   parameters['loss']['class_name'], 
                                   parameters['loss']['parameters'])

        # VAL METRICS
        self.metrics = []
        for metric_config in parameters['metric']:
            metric_instance = instanciate_module(metric_config['module_name'],
                                   metric_config['class_name'], 
                                   metric_config['parameters'])
            self.metrics.append({'name': metric_config['class_name'], 'func': metric_instance})
        
    def train(self, dl: DataLoader):
        raise NotImplementedError
    
    def test(self, dl: DataLoader):
        raise NotImplementedError
    
    def fit(self, train_dl, test_dl, log_dir: str):
        num_epochs = self.parameters['num_epochs']
        for epoch in range(num_epochs):
            train_loss = self.train(train_dl)
            test_loss, all_preds, all_targets = self.test(test_dl)
            test_metrics = [{'name': metric['name'], 'value': metric['func'](all_preds, all_targets)} for metric in self.metrics]

            if self.parameters['track']:
                wandb.log({
                    f"Train/{self.parameters['loss']['class_name']}": train_loss,
                    f"Test/{self.parameters['loss']['class_name']}": test_loss,
                    "_step_": epoch
                })
                for metric in test_metrics:
                    wandb.log({f"Test/{metric['name']}": metric['value'], '_step_': epoch})
                
            if self.lr_scheduler is not None:
                self.lr_scheduler.step(test_loss)

            self.early_stop(self.model, test_loss, log_dir, epoch)

            epoch_log_message = f"Epoch {epoch + 1} / {num_epochs} - Train/Test {self.parameters['loss']['class_name']}: {train_loss:.4f} | {test_loss:.4f}"

            for metric in test_metrics:
                epoch_log_message += f"; {metric['name']}: {metric['value']:.4f}"

            logging.info(epoch_log_message)

            if self.early_stop.stop:
                logging.info(
                    f"Val loss did not improve for {self.early_stop.patience} epochs.")
                logging.info('Training stopped by early stopping mecanism.')
                break
            
        if self.parameters['track']:
            wandb.finish()
