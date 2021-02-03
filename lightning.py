#%%
from numpy.core.numeric import cross
from pytorch_lightning import callbacks
from pytorch_lightning.callbacks import early_stopping
from pytorch_lightning.core.step_result import weighted_mean
from sklearn import model_selection
from sklearn.utils import shuffle
import torch
from torch._C import device
from torch.nn import functional as F
import torch.nn as nn
import pytorch_lightning as pl
from torchvision.models import resnet34
from torchvision import transforms, datasets
# クロスバリデーション
from sklearn.model_selection import KFold
from torchvision.transforms.transforms import ColorJitter, Resize
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

class ClassNet(pl.LightningModule):

    def __init__(self):
        super().__init__()

        resnet = resnet34(pretrained=True)
        # print(resnet.sum)
        self.conv = resnet
        for param in self.conv.parameters():
            param.requires_grad = False

        self.c4 = nn.Sequential(
            nn.BatchNorm1d(1000),
            nn.Linear(1000, 32),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.20799628466288428),
            nn.Linear(32, 368),
            nn.ReLU(inplace=True),
            nn.Linear(368, 10)            
        )

        self.train_acc = pl.metrics.Accuracy()
        self.val_acc = pl.metrics.Accuracy()
        self.test_acc = pl.metrics.Accuracy()

    def forward(self, x):
        batch_size, channels, width, height = x.size()
        x = self.conv(x)
        x = self.c4(x)
        x = F.log_softmax(x, dim=1)
        return x

    def training_step(self, batch, batch_idx):
        # print(len(batch))

        x, t = batch
        y = self(x) # forward
        loss = F.cross_entropy(y, t)

        # log
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        self.log('train_acc', self.train_acc(y, t), on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, t = batch
        y = self(x)
        loss = F.cross_entropy(y, t)

        # log
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        self.log('val_acc', self.val_acc(y, t), on_step=False, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        x, t = batch
        y = self(x)
        loss = F.cross_entropy(y, t)

        # log
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.log('test_acc', self.val_acc(y, t), on_step=False, on_epoch=True)

        return loss

    # 最適化
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.07369726824451693, weight_decay=0.0002097517651377327)
        return optimizer

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    ])

dataset = datasets.ImageFolder('./train_sorted', transform)

esc = EarlyStopping(
    min_delta=0.00,
    patience=1,
    verbose=False,
    monitor='val_loss',
    mode='min',
)

# クロスバリデーション
# batch_size = 128

# n_trainval = int(len(dataset) * 0.8)
# n_test = len(dataset) - n_trainval
# trainval, test = torch.utils.data.random_split(dataset, [n_trainval, n_test])
# test_loader = torch.utils.data.DataLoader(test, batch_size, shuffle=False, num_workers=16)

# net = ClassNet()

# kf = KFold(n_splits=5, shuffle=True)
# for train_index, val_index in kf.split(trainval):

#     train_data = torch.utils.data.dataset.Subset(dataset, train_index)
#     train_loader = torch.utils.data.DataLoader(train_data, batch_size, shuffle=True, drop_last=True, num_workers=16)

#     val_data = torch.utils.data.dataset.Subset(dataset, val_index)
#     val_loader = torch.utils.data.DataLoader(val_data, batch_size, shuffle=False, num_workers=16)

#     trainer = pl.Trainer(max_epochs=20, gpus=1, callbacks=[esc])
#     trainer.fit(net, train_loader, val_loader)
#     trainer.callback_metrics

#     trainer.test(test_dataloaders=test_loader)


# クロスバリデーションなし
n_train = int(len(dataset) * 0.7)
n_val = len(dataset) - n_train
n_test = n_val
train, val = torch.utils.data.random_split(dataset, [n_train, n_val])

batch_size = 110

train_loader = torch.utils.data.DataLoader(train, batch_size, shuffle=True, drop_last=True, num_workers=16)
val_loader = torch.utils.data.DataLoader(val, batch_size, shuffle=False, num_workers=16)

net = ClassNet()

trainer = pl.Trainer(max_epochs=100, gpus=1, callbacks=[esc])

trainer.fit(net, train_loader, val_loader)

trainer.callback_metrics

#%%
net.eval()
net.freeze()

import glob
from PIL import Image
import re
import os

paths = glob.glob('test_images/*.jpg')

ptn = re.compile(r'\d+')
files = sorted(paths, key=lambda s: int(ptn.search(s).group()))

with open('submit.tsv', 'w') as f:

    for file in files:
        img = Image.open(file)
        x = transform(img).unsqueeze(0)
        # x = x.to("cuda")
        pred = net(x)
        f.write("{}\t{}\n".format(os.path.basename(file), pred.argmax()))

# %%
