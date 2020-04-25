from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
import torch
from torch import nn
from model import UNetModel
from dataset import UNetDataset
from transform import UNetTransform
from torch.utils.data import DataLoader

class UNetSystem(pl.LightningModule):
    def __init__(self, dataset_path, criteria, in_channel, num_class, batch_size):
        super(UNetSystem, self).__init__()
        self.dataset_path = dataset_path
        self.model = UNetModel(in_channel, num_class)
        self.criteria = criteria
        self.batch_size = batch_size

    def forward(self, x):
        x = self.model(x)

        return x

    def training_step(self, batch, batch_idx):
        image, label = batch
        y_pred = self.forward(image)
        loss = nn.functional.cross_entropy(y_pred, label)
        tensorboard_logs = {"train_loss" : loss }
        
        return {"loss" : loss, "log" : tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        image, label = batch
        out = self.forward(image)
        loss = nn.functional.cross_entropy(out, label)
        return {"val_loss" : loss}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())

        return optimizer

    @pl.data_loader
    def train_dataloader(self):
        translate = 0
        rotate = 360
        shear = 0
        scale = 0.05
        batch_size = 15

        train_dataset = UNetDataset(
                dataset_path = self.dataset_path, 
                phase = "train", 
                criteria = self.criteria,
                transform = UNetTransform(translate, rotate, shear, scale)
                )

        train_loader = DataLoader(
                train_dataset ,
                shuffle=True, 
                batch_size = self.batch_size
                )

        return train_loader

    @pl.data_loader
    def val_dataloader(self):
        val_dataset = UNetDataset(
                dataset_path = self.dataset_path, 
                phase = "val", 
                criteria = self.criteria,
                transform = UNetTransform()
                )

        val_loader = DataLoader(
                val_dataset, 
                batch_size = self.batch_size
                )

        return val_loader








