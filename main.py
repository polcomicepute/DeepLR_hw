import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from IPython.core.display import display
# from pl_bolts.datamodules import CIFAR10DataModule
# from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim.swa_utils import AveragedModel, update_bn
from torchmetrics.functional import accuracy

from torch.utils.data import DataLoader, random_split
from torchvision import transforms 
import pytorch_lightning as pl

seed_everything(7)
wandb_logger = WandbLogger(name='Resnet_Adam_0.001_256',project='2022707003_이혜민_pytorchlightning_Cifar')
from torchvision.datasets import CIFAR10
PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
BATCH_SIZE = 256 if torch.cuda.is_available() else 64
NUM_WORKERS = int(os.cpu_count() / 4)

# train_transforms = torchvision.transforms.Compose(
#     [
#         torchvision.transforms.RandomCrop(32, padding=4),
#         torchvision.transforms.RandomHorizontalFlip(),
#         torchvision.transforms.ToTensor(),
#         cifar10_normalization(),
#     ]
# )

# test_transforms = torchvision.transforms.Compose(
#     [
#         torchvision.transforms.ToTensor(),
#         cifar10_normalization(),
#     ]
# )

class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.mean = (0.4914, 0.4822, 0.4465)
        self.std = (0.2471, 0.2435, 0.2616)
        self.data_dir = PATH_DATASETS
        self.train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        self.dims = (3, 32, 32)
        self.num_classes = 10

    def prepare_data(self):
            # download
            CIFAR10(self.data_dir, train=True, download=True)
            CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            cifar_full = CIFAR10(self.data_dir, train=True, transform=self.transform)
            self.cifar_train, self.cifar_val = random_split(cifar_full, [45000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.cifar_test = CIFAR10(self.data_dir, train=False, transform=self.transform)
    def train_dataloader(self):
            return DataLoader(self.cifar_train, batch_size=BATCH_SIZE//4, num_workers= NUM_WORKERS)
    
    def val_dataloader(self):
            return DataLoader(self.cifar_val, batch_size=BATCH_SIZE//4, num_workers= NUM_WORKERS)

    def test_dataloader(self):
        return DataLoader(self.cifar_test, batch_size=BATCH_SIZE//4, num_workers= NUM_WORKERS)

cifar10_dm = CIFAR10DataModule()
# STEP_PER_EPOCH =len(cifar10_dm.train_dataloader)
# print('STEP_PER_EPOCH', STEP_PER_EPOCH)
def create_model():
    model = torchvision.models.resnet18(pretrained=False, num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.maxpool = nn.Identity()
    return model

class LitResnet(LightningModule):
    def __init__(self, lr=0.001):
        super().__init__()

        self.save_hyperparameters()
        self.model = create_model()

    def forward(self, x):
        out = self.model(x)
        return F.log_softmax(out, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.log("train_loss", loss)
        return loss

    def evaluate(self, batch, stage=None):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_acc", acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

        # def configure_optimizers(self):
        #         optimizer = torch.optim.SGD(
        #             self.parameters(),
        #             lr=self.hparams.lr,
        #             momentum=0.9,
        #             weight_decay=5e-4,
        #         )
        #         steps_per_epoch = 45000 // BATCH_SIZE
        #         scheduler_dict = {
        #             "scheduler": OneCycleLR(
        #                 optimizer,
        #                 0.1,
        #                 epochs=self.trainer.max_epochs,
        #                 steps_per_epoch=steps_per_epoch,
        #             ),
        #             "interval": "step",
        #         }
        #         return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr
        )
        # scheduler_dict = {
        #     "scheduler": OneCycleLR(
        #         optimizer,
        #         0.1,
        #         epochs=self.trainer.max_epochs,
        #         total_steps=self.trainer.max_epochs*BATCH_SIZE,
        #     ),
        #     "interval": "step",
        # }
        # return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}
        # return {"optimizer": optimizer}
        # steps_per_epoch = 45000 // BATCH_SIZE
        scheduler_dict = {
            "scheduler": OneCycleLR(
                optimizer,
                0.1,
                epochs=self.trainer.max_epochs,
                # steps_per_epoch=steps_per_epoch,
                total_steps=self.trainer.max_epochs*BATCH_SIZE,

            ),
            "interval": "step",
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}

# model = LitResnet(lr=0.001)
model = LitResnet(lr=0.005)


# trainer = Trainer(
#     max_epochs=30,
#     accelerator="gpu",
#     devices=4 if torch.cuda.is_available() else None,  # limiting got iPython runs
#     logger=wandb_logger,
#     strategy='ddp',
#     callbacks=[LearningRateMonitor(logging_interval="step"), TQDMProgressBar(refresh_rate=10)],
# )
trainer = Trainer(
    max_epochs=30,
    accelerator="gpu",
    devices=4, 
    logger=wandb_logger,
    strategy='ddp',
    callbacks=[LearningRateMonitor(logging_interval="step"), TQDMProgressBar(refresh_rate=10)],
)

# trainer.fit(model=model, train_dataloaders = cifar10_dm.train_dataloader,val_dataloaders = cifar10_dm.val_dataloader)
# trainer.test(model=model, datamodule=cifar10_dm.test_dataloader())
trainer.fit(model, cifar10_dm)
trainer.test(model, cifar10_dm)
