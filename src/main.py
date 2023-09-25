import yaml
from pathlib import Path
import inspect
from typing import Any, Optional
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
from pytorch_lightning import Trainer
import importlib
import pytorch_lightning.callbacks as plc
from torch.utils.data import DataLoader
from torch.nn import functional as F
import torch.optim.lr_scheduler as lrs
import pytorch_lightning as pl
from sklearn.svm import LinearSVC, SVC
import argparse
from pytorch_lightning.loggers import WandbLogger

from utils import Evaluator, EmbeddingEvaluation


class Train(pl.LightningModule):
    def __init__(self, **kargs):
        super().__init__()
        self.save_hyperparameters()
        self.load_model()

    def training_step(self, batch, batch_idx):
        batch = batch[0]
        out = self.model(batch)
        loss = self.model.loss(out, batch)
        metrics = self.metrics(out, batch)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=self.hparams.batch_size)
        for k, v in metrics.items():
            self.log(f'train_{k}', v, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.hparams.batch_size)
        return loss
    
    def validation_step(self, batch, batch_idx):
        batch = batch[0]
        out = self.model(batch)
        loss = self.model.loss(out, batch)
        metrics = self.metrics(out, batch)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.hparams.batch_size)
        for k, v in metrics.items():
            self.log(f'val_{k}', v, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.hparams.batch_size)
        return loss
    
    def test_step(self, batch, batch_idx):
        batch = batch[0]
        out = self.model(batch)
        loss = self.model.loss(out, batch)
        metrics = self.metrics(out, batch)
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.hparams.batch_size)
        for k, v in metrics.items():
            self.log(f'test_{k}', v, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.hparams.batch_size)
        return loss
    
    def configure_optimizers(self):
        if hasattr(self.hparams, 'weight_decay'):
            weight_decay = self.hparams.weight_decay
        else:
            weight_decay = 0
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=weight_decay)
        if self.hparams.lr_scheduler is None:
            return optimizer
        else:
            if self.hparams.lr_scheduler == 'step':
                scheduler = lrs.StepLR(optimizer,
                                       step_size=self.hparams.lr_decay_steps,
                                       gamma=self.hparams.lr_decay_rate)
            elif self.hparams.lr_scheduler == 'cosine':
                scheduler = lrs.CosineAnnealingLR(optimizer,
                                                  T_max=self.hparams.lr_decay_steps,
                                                  eta_min=self.hparams.lr_decay_min_lr)
            else:
                raise ValueError('Invalid lr_scheduler type!')
            return [optimizer], [scheduler]

    def load_model(self):
        name = self.hparams.model_name
        upper_name = name.upper()
        try:
            Model = getattr(importlib.import_module('models.'+name, package=__package__), upper_name)
        except:
            raise ValueError(f'Invalid Module File Name or Invalid Class Name {name}.{upper_name}!')
        self.model = self.instancialize(Model)

    def instancialize(self, Model, **other_args):
        class_args = inspect.getargspec(Model.__init__).args[1:]
        inkeys = self.hparams.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = getattr(self.hparams, arg)
        args1.update(other_args)
        return Model(**args1)
    
    def metrics(self, out, batch):
        pass

class DInterface(pl.LightningDataModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.load_data_module()
        
    def setup(self, stage):
        if stage == 'fit' or stage is None:
            self.trainset = self.instancialize(train=True)
            self.valset = self.instancialize(train=False)
        if stage == 'test' or stage is None:
            self.testset = self.instancialize(train=False)

    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, shuffle=True, persistent_workers=True)
    
    def val_dataloader(self):
        return DataLoader(self.valset, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, shuffle=False, persistent_workers=True)
    
    def test_dataloader(self):
        return DataLoader(self.testset, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, shuffle=False, persistent_workers=True)
    
    def load_data_module(self):
        name = self.hparams.dataset_name
        camel_name = ''.join([i.capitalize() for i in name.split('_')])
        try:
            self.data_module = getattr(importlib.import_module('.'+name, package=__package__), camel_name)
        except:
            raise ValueError(f'Invalid Dataset File Name or Invalid Class Name data.{name}.{camel_name}')
        
    def instancialize(self, **other_args):
        class_args = inspect.getargspec(self.data_module.__init__).args[1:]
        inkeys = self.hparams.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = self.hparams[arg]
        args1.update(other_args)
        return self.data_module(**args1)
    
    def get_in_out_dim(self):
        ins = self.instancialize()
        feat_dim = ins.dataset[0].x.shape[1]
        labels = [int(data.y.cpu().numpy()) for data in ins.dataset]
        class_num = len(set(labels))
        print(f'Feature dimension: {self.feat_dim}')
        print(f'Number of classes: {self.class_num}')
        return feat_dim, class_num

def load_callbacks(args):
    callbacks = []
    callbacks.append(plc.EarlyStopping(
        monitor='train_loss',
        mode='min',
        patience=10,
        min_delta=0.01
    ))

    callbacks.append(plc.ModelCheckpoint(
        monitor='train_loss',
        filename='best-{epoch:02d}-{train_loss:.3f}',
        save_top_k=1,
        mode='min',
        save_last=True
    ))

    if args.lr_scheduler:
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
    args.in_dim, args.num_classes = feat_dim, class_num

    model = Train(**vars(args))
    wandb_logger = WandbLogger(project=args.project, save_dir=args.log_dir)
    args.callbacks = load_callbacks(args)
    trainer = Trainer(max_epochs=args.epochs, accelerator='gpu',\
                          log_every_n_steps=1, logger=wandb_logger, callbacks=args.callbacks)
    trainer.fit(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module)

if __name__ == "__main__":
    main()


