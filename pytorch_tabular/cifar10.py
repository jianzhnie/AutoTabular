import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from pl_bolts.datamodules import CIFAR10DataModule
from pl_bolts.callbacks import PrintTableMetricsCallback
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
from pytorch_lightning import LightningModule, seed_everything, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, MLFlowLogger
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim.swa_utils import AveragedModel, update_bn
from torchmetrics.functional import accuracy
import sys
sys.path.append("../../")
from autotorch.models.network import init_network
from autotorch.autoptl.custom_trainer import CustomTrainer

seed_everything(7)

PATH_DATASETS = os.environ.get(
    '/media/robin/DATA/datatsets/image_data/cifar10')
AVAIL_GPUS = min(1, torch.cuda.device_count())
BATCH_SIZE = 16 if AVAIL_GPUS else 32
NUM_WORKERS = int(os.cpu_count() / 2)

train_transforms = torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop(32),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    cifar10_normalization(),
])

test_transforms = torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop(32),
    torchvision.transforms.ToTensor(),
    cifar10_normalization(),
])

cifar10_dm = CIFAR10DataModule(
    data_dir=PATH_DATASETS,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    train_transforms=train_transforms,
    test_transforms=test_transforms,
    val_transforms=test_transforms,
)

# Data loading code
root_dir = '/media/robin/DATA/datatsets/image_data/shopee-iet/images'
traindir = os.path.join(root_dir, 'train')
valdir = os.path.join(root_dir, 'val')
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

train_dataset = datasets.ImageFolder(
    traindir,
    transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]))

train_sampler = None
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=32,
                                           shuffle=(train_sampler is None),
                                           num_workers=0,
                                           pin_memory=True,
                                           sampler=train_sampler)

val_dataset = datasets.ImageFolder(
    valdir,
    transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]))

val_loader = torch.utils.data.DataLoader(val_dataset,
                                         batch_size=32,
                                         shuffle=False,
                                         num_workers=0,
                                         pin_memory=True)


def create_model(model_name):
    model = torchvision.models.resnet18(pretrained=False, num_classes=4)
    # model = init_network(model_name, num_class=10, pretrained=True)
    # model.conv1 = nn.Conv2d(3,
    #                         64,
    #                         kernel_size=(3, 3),
    #                         stride=(1, 1),
    #                         padding=(1, 1),
    #                         bias=False)
    # model.maxpool = nn.Identity()
    return model


class LitResnet(LightningModule):
    def __init__(self, lr=0.05):
        super().__init__()

        self.save_hyperparameters()
        self.model = create_model('resnet18')

    def forward(self, x):
        out = self.model(x)
        return F.log_softmax(out, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)

        # logs metrics for each training_step,
        # and the average across the epoch, to the progress bar and logger
        self.log('train_loss',
                 loss,
                 on_step=True,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True,
                 sync_dist=True)
        return loss

    def evaluate(self, batch, stage=None):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)

        if stage:
            self.log(f'{stage}_loss', loss, prog_bar=True)
            self.log(f'{stage}_acc', acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, 'val')

    # def test_step(self, batch, batch_idx):
    #     self.evaluate(batch, 'test')

    def test_step(self, batch, batch_idx):
        x, y = batch
        # implement your own
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)
        # log the outputs!
        self.log_dict({'test_loss': loss, 'test_acc': acc})

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.lr,
            momentum=0.9,
            weight_decay=5e-4,
        )
        steps_per_epoch = 45000 // BATCH_SIZE
        scheduler_dict = {
            'scheduler':
            OneCycleLR(
                optimizer,
                0.1,
                epochs=self.trainer.max_epochs,
                steps_per_epoch=steps_per_epoch,
            ),
            'interval':
            'step',
        }

        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler_dict,
        }

    def configure_callbacks(self):
        checkpoint = ModelCheckpoint(monitor="val_loss")
        return [checkpoint]


class PrintCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        print("Training is started!")

    def on_train_end(self, trainer, pl_module):
        print("Training is done.")


early_stop_callback = EarlyStopping(monitor='val_acc',
                                    min_delta=0.00,
                                    patience=3,
                                    verbose=False,
                                    mode='max')

model = LitResnet(lr=0.05)
model.datamodule = cifar10_dm
trainer = CustomTrainer(
    progress_bar_refresh_rate=50,
    log_every_n_steps=1,
    log_gpu_memory='all',
    max_epochs=10,
    gpus=AVAIL_GPUS,
    sync_batchnorm=True,
    limit_train_batches=1.0,
    checkpoint_callback=True,
    check_val_every_n_epoch=1,
    precision=16,
    profiler="simple",
    val_check_interval=1.0,
    weights_summary='top',
    auto_scale_batch_size=True,
    benchmark=True,
    weights_save_path='lightning_logs/',
    default_root_dir=os.getcwd(),
    max_time={
        "days": 1,
        "hours": 5
    },
    logger=[
        TensorBoardLogger(save_dir='lightning_logs/',
                          version="0",
                          name='resnet'),
        MLFlowLogger(save_dir='mlflow_logs/')
    ],
    callbacks=[
        LearningRateMonitor(logging_interval='step'),
        PrintTableMetricsCallback(),
        early_stop_callback,
    ],
)

# trainer.fit(model, cifar10_dm)
# trainer.test(model, datamodule=cifar10_dm)

trainer.fit(model, train_dataloader=train_loader,  val_dataloaders=val_loader)
trainer.test(model, val_loader)

