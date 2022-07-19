# 1. Importing modules
import os
from random import random
import pandas as pd
import numpy as np
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
# from torch.utils.data import DataLoader, RandomSampler, Subset
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
from data.puppy.CustImgDataset import PuppyImageDataset

gc.collect()
torch.cuda.empty_cache()

torch.manual_seed(7866)    # https://pytorch.org/docs/stable/notes/randomness.html
np.random.seed(7866)

# 2. Intializing the hyperparamters
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print(torch.__version__, DEVICE)
BATCH_SIZE = 100
EPOCHS = 10

num_breeds = 120    # Requires generalization
train_prop = 0.8

# 3. Loading the puppy image dataset
my_transform = transforms.Compose([     # Augmentation methods should go into this: http://incredible.ai/pytorch/2020/04/25/Pytorch-Image-Augmentation/
  transforms.Resize([128, 128])])
  # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
  # transforms.ToTensor()])
my_target_transform = \
  transforms.Lambda(lambda y: torch.zeros(num_breeds, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))

train_ds = PuppyImageDataset(annotations_file='.\\data\\puppy\\annotation_input.csv',
                                  img_dir='.\\data\\puppy\\train',
                                  transform=my_transform,
                                  target_transform=my_target_transform)

train_size = int(len(train_ds)*train_prop)
val_size = len(train_ds) - train_size

pseudo_train_ds, pseudo_val_ds = \
  random_split(train_ds, [train_size, val_size], generator=torch.Generator().manual_seed(7866))
# sample_image, sample_label = pseudo_train_ds.__getitem__(0)
# plt.imshow(sample_image.permute(1, 2, 0))
pseudo_train_dl = DataLoader(dataset=pseudo_train_ds, batch_size=BATCH_SIZE, shuffle=True)
pseudo_val_dl = DataLoader(dataset=pseudo_val_ds, batch_size=BATCH_SIZE, shuffle=False)

# Checking data 
for(x_train, y_train) in pseudo_train_dl:
  print('x_train: {0}, type: {1}'.format(x_train.size(), x_train.type()))
  print('y_train: {0}, type: {1}'.format(y_train.size(), y_train.type()))

  plt.figure(figsize=(20, 2))
  for i in range(10):
    plt.subplot(1, 10, i+1)
    plt.axis('off')
    plt.imshow(x_train[i].permute(1, 2, 0))
    plt.title('Class: {0}'.format(str(y_train[i].argmax().item())))
  break

def make_sequential_x2(in_channels, out_channels, kernel_size, p):
  sequential = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, padding=1), 
    nn.BatchNorm2d(out_channels), nn.ReLU(),
    nn.Conv2d(out_channels, out_channels, kernel_size, padding=1),
    nn.BatchNorm2d(out_channels), nn.ReLU(), nn.Dropout2d(p=p), nn.MaxPool2d(kernel_size=2, stride=2))
  return sequential

def make_sequential_x3(in_channels, out_channels, kernel_size, p):
  sequential = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, padding=1), 
    nn.BatchNorm2d(out_channels), nn.ReLU(),
    nn.Conv2d(out_channels, out_channels, kernel_size, padding=1),
    nn.BatchNorm2d(out_channels), nn.ReLU(), 
    nn.Conv2d(out_channels, out_channels, kernel_size, padding=1),
    nn.BatchNorm2d(out_channels), nn.ReLU(), nn.Dropout2d(p=p), nn.MaxPool2d(kernel_size=2, stride=2))
  return sequential

class CNN(nn.Module):
  def __init__(self):
    super(CNN, self).__init__()    # Same with the class name?
    self.fl1 = make_sequential_x2(3, 64, 3, 0.25)    # (64, 64, 64)
    self.fl2 = make_sequential_x2(64, 128, 3, 0.25)    # (32, 32, 128)
    self.fl3 = make_sequential_x3(128, 256, 3, 0.25)    # (16, 16, 256)
    self.fl4 = make_sequential_x3(256, 512, 3, 0.25)    # (8, 8, 512)
    self.aa = nn.AdaptiveAvgPool2d(2) #(2, 2, 512)
    self.fcs = nn.Sequential(nn.Linear(2048, 2048),
      nn.ReLU(),
      nn.Dropout(),
      nn.Linear(2048, 2048),
      nn.ReLU(),
      nn.Dropout(),
      nn.Linear(2048, num_breeds))
  def forward(self, x):
    x = self.fl1(x)
    x = self.fl2(x)
    x = self.fl3(x)
    x = self.fl4(x)
    x = self.aa(x)
    x = x.view(-1, 512*2*2)
    x = self.fcs(x)
    x = F.log_softmax(x, dim=0)
    return x

model = CNN().to(DEVICE)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

torch.cuda.empty_cache()

def train(model, train_loader, optimizer, log_interval):
  n = len(train_loader.dataset)
  model.train()
  for batch_idx, (x_train, y_train) in enumerate(train_loader):
    x_train = x_train.to(DEVICE, dtype=torch.float32)
    y_train = y_train.to(DEVICE)
    optimizer.zero_grad()
    output = model(x_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()
    
    if batch_idx % log_interval == 0:
      print("Train epoch: {} [{}/{}({:.1f}%)]\tTrain loss: {:.6f}".format(epoch, batch_idx * BATCH_SIZE, n, 100*BATCH_SIZE*batch_idx/n, loss.item()))

def evaluate(model, val_loader):
  model.eval()
  val_loss = 0
  correct = 0
  with torch.no_grad():
    for x_val, y_val in val_loader:
      x_val = x_val.to(DEVICE, dtype=torch.float32)
      y_val = y_val.to(DEVICE)
      output = model(x_val)
      val_loss += criterion(output, y_val).item()
      prediction = output.argmax(dim=1)
      correct += (prediction == y_val.argmax(dim=1)).sum().item()
  val_loss /= len(val_loader.dataset)
  val_accuracy = 100. * correct / len(val_loader.dataset)
  return val_loss, val_accuracy

for epoch in range(1, EPOCHS + 1):
  train(model, pseudo_train_dl, optimizer, log_interval=5)
  val_loss, val_accuracy = evaluate(model, pseudo_val_dl)
  print('\n[Epoch: {}], \tTest loss: {:.4f}, \tTest accuracy: {:.2f}% \n'.format(epoch, val_loss, val_accuracy))



# 999. Trashdddddddd

check_img_load = next(iter(pseudo_train_dl))[0]
plt.imshow(check_img_load[2, :, :, :].permute(1, 2, 0))

train_dl = DataLoader(dataset=train_ds, batch_size=BATCH_SIZE, shuffle=True)
len(train_dl)
len(train_dl.dataset)

check_img_load = next(iter(train_dl))[0]
plt.imshow(check_img_load[2, :, :, :].permute(1, 2, 0))

test_index = 12
sample_image, sample_label = train_dataset.__getitem__(test_index)

# pseudo_train_ds = Subset(train_ds, np.arange(7500))
# pseudo_train_dl = DataLoader(pseudo_train_ds, sampler=RandomSampler(pseudo_train_ds), batch_size=BATCH_SIZE)

# iter_train_loader = iter(train_loader)
# type(train_loader)
# type(iter_train_loader)

# breed_index = train_dataset.img_labels.iloc[test_index, 1]
# breed_text = pd.read_csv('breed_matching_table.csv').iloc[breed_index, 0]
# plt.imshow(sample_image.permute(1, 2, 0))
# plt.title(breed_text)



