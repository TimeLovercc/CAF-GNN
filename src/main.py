import argparse
import importlib
import inspect
import yaml
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.optim.lr_scheduler as lrs
import torch_geometric.transforms as T
import pytorch_lightning as pl
import pytorch_lightning.callbacks as plc
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from torch_geometric.loader import DataLoader
from torchmetrics.functional.classification import (
    binary_accuracy,
    binary_auroc,
    binary_f1_score,
    multiclass_accuracy,
    multiclass_auroc,
    multiclass_f1_score,
)

torch.set_float32_matmul_precision('medium')

class Train(pl.LightningModule):
    def __init__(self, **kargs):
        super().__init__()
        self.save_hyperparameters()
        self.load_model()

    def training_step(self, batch, batch_idx):
        batch = batch[0]
        out = self.model(batch)
        loss = self.model.loss(out, batch, mode='train')
        metrics = self.metrics(out, batch, mode='train')
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=False, batch_size=1)
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=False, batch_size=1)
        return loss
    
    def validation_step(self, batch, batch_idx):
        batch = batch[0]
        out = self.model(batch)
        loss = self.model.loss(out, batch, mode='val')
        metrics = self.metrics(out, batch, mode='val')
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=1)
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=False, batch_size=1)
        return loss
    
    def test_step(self, batch, batch_idx):
        batch = batch[0]
        out = self.model(batch)
        loss = self.model.loss(out, batch, mode='test')
        metrics = self.metrics(out, batch, mode='test')
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=False, batch_size=1)
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=False, batch_size=1)
        return loss
    
    def configure_optimizers(self):
        if 'weight_decay' in  self.hparams.model_config.keys():
            weight_decay = self.hparams.model_config['weight_decay']
        else:
            weight_decay = 0
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.model_config['lr'], weight_decay=weight_decay)
        if self.hparams.model_config['lr_scheduler'] == None:
            return optimizer
        else:
            if self.hparams.model_config['lr_scheduler'] == 'step':
                scheduler = lrs.StepLR(optimizer,
                                       step_size=self.hparams.model_config['lr_decay_steps'],
                                       gamma=self.hparams.model_config['lr_decay_rate'])
            elif self.hparams.model_config['lr_scheduler'] == 'cosine':
                scheduler = lrs.CosineAnnealingLR(optimizer,
                                                  T_max=self.hparams.model_config['lr_decay_steps'],
                                                  eta_min=self.hparams.model_config['lr_decay_min_lr'])
            else:
                raise ValueError('Invalid lr_scheduler type!')
            return [optimizer], [scheduler]

    def load_model(self):
        name = self.hparams.model_name
        upper_name = name.upper()
        try:
            sys.path.append('./src/models')
            Model = getattr(importlib.import_module(name), upper_name)
        except:
            raise ValueError(f'Invalid Module File Name or Invalid Class Name {name}.{upper_name}!')
        self.model = self.instancialize(Model)

    def instancialize(self, Model, **other_args):
        class_args = inspect.getfullargspec(Model.__init__).args[1:]
        inkeys = self.hparams.model_config.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = self.hparams.model_config[arg]
        args1.update(other_args)
        return Model(**args1)
    
    def metrics(self, out, batch, mode):
        preds = out
        labels, sens, mask = batch['y'], batch['sens'], batch[f'{mode}_mask']
        if self.hparams.model_config['out_dim'] == 1:
            acc = binary_accuracy(preds[mask], labels[mask])
            f1 = binary_f1_score(preds[mask], labels[mask])
            auroc = binary_auroc(preds[mask], labels[mask])
        elif self.hparams.model_config['out_dim'] > 2:
            acc = multiclass_accuracy(preds[mask], labels[mask], num_classes=self.hparams.out_dim, average='micro')
            f1 = multiclass_f1_score(preds[mask], labels[mask], num_classes=self.hparams.out_dim, average='micro')
            auroc = multiclass_auroc(preds[mask], labels[mask], num_classes=self.hparams.out_dim, average='macro')
        parity, equality = self.binary_fair_metrics(preds[mask], labels[mask], sens[mask])
        fair = acc + f1 + auroc - self.hparams.alpha * (parity + equality)
        return {f'{mode}_acc': acc, f'{mode}_f1': f1, f'{mode}_auroc': auroc, f'{mode}_parity': parity, \
                f'{mode}_equality': equality, f'{mode}_fair': fair}
    
    def binary_fair_metrics(self, preds, labels, sens):
        idx_s0 = sens==0
        idx_s1 = sens==1
        idx_s0 = idx_s0.detach().cpu().numpy()
        idx_s1 = idx_s1.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        idx_s0_y1 = np.bitwise_and(idx_s0, labels==1)
        idx_s1_y1 = np.bitwise_and(idx_s1, labels==1)
        preds = (preds.squeeze()>0.5)
        parity = abs(sum(preds[idx_s0])/sum(idx_s0)-sum(preds[idx_s1])/sum(idx_s1))
        equality = abs(sum(preds[idx_s0_y1])/sum(idx_s0_y1)-sum(preds[idx_s1_y1])/sum(idx_s1_y1))
        return parity.item(), equality.item()

class DInterface(pl.LightningDataModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.load_data_module()
        
    def setup(self, stage):
        if stage == 'fit' or stage is None:
            self.trainset = self.dataset
            self.valset = self.dataset
        if stage == 'test' or stage is None:
            self.testset = self.dataset

    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=1, shuffle=False)
    
    def val_dataloader(self):
        return DataLoader(self.valset, batch_size=1, shuffle=False)
    
    def test_dataloader(self):
        return DataLoader(self.testset, batch_size=1, shuffle=False)
    
    def load_data_module(self):
        name = self.hparams.dataset_name
        camel_name = ''.join([i.capitalize() for i in name.split('_')])
        try:
            sys.path.append('./src/datasets')
            Dataset = getattr(importlib.import_module(name), camel_name)
        except:
            raise ValueError(f'Invalid Dataset File Name or Invalid Class Name {name}.{camel_name}')
        self.dataset = self.instancialize(Dataset)
        
    def instancialize(self, Dataset, **other_args):
        class_args = inspect.getfullargspec(Dataset.__init__).args[1:]
        inkeys = self.hparams.data_config.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = self.hparams.data_config[arg]
        args1.update(other_args)
        if args1['transform'] == 'normalize':
            args1['transform'] = T.NormalizeFeatures()
        return Dataset(**args1)
    
    def get_in_out_dim(self):
        feat_dim = self.dataset.num_features
        class_num = self.dataset.num_classes
        print(f'Feature dimension: {feat_dim}')
        print(f'Number of classes: {class_num}')
        return feat_dim, class_num

def load_callbacks(args):
    callbacks = []
    # callbacks.append(plc.EarlyStopping(
    #     monitor='val_loss',
    #     mode='min',
    #     patience=30,
    #     min_delta=1
    # ))

    callbacks.append(plc.ModelCheckpoint(
        monitor='val_loss',
        filename='best',
        save_top_k=1,
        mode='min',
        save_last=True
    ))

    callbacks.append(plc.RichProgressBar(
        refresh_rate=1
    ))

    if args.model_config['lr_scheduler'] != None:
        callbacks.append(plc.LearningRateMonitor(
            logging_interval='epoch'))
    return callbacks

def main():
    parser = argparse.ArgumentParser(description='CAF')
    parser.add_argument('--dataset_name', type=str, help='dataset name')
    parser.add_argument('--model_name', type=str, help='model name')
    parser.add_argument('--seed', type=int, help='random seed')
    args = parser.parse_args()
    config_dir = Path('./src/configs')
    global_config = yaml.safe_load((config_dir / 'global_config.yml').open('r'))
    local_config = yaml.safe_load((config_dir / f'{args.dataset_name}_{args.model_name}.yml').open('r'))
    args = argparse.Namespace(**{**global_config, **local_config, **vars(args)})
    
    pl.seed_everything(args.seed)
    data_module = DInterface(**vars(args))
    feat_dim, class_num = data_module.get_in_out_dim()
    args.model_config['in_dim'] = feat_dim
    args.model_config['out_dim'] = class_num if class_num > 2 else 1

    model = Train(**vars(args))
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    csv_logger = CSVLogger(save_dir=Path(args.log_dir) / f'{args.dataset_name}_{args.model_name}', version=timestamp)
    callbacks = load_callbacks(args)
    trainer = Trainer(max_epochs=args.epochs, accelerator='gpu',\
                          logger=csv_logger, log_every_n_steps=1, callbacks=callbacks)
    trainer.fit(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module)

    if args.model_name == 'caf':
        model.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
        trainer.fit(model, datamodule=data_module)
        trainer.test(model, datamodule=data_module)

if __name__ == "__main__":
    main()


