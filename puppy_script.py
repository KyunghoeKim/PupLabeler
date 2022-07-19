
# 1. Importing modules
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from data.puppy.CustImgDataset import PuppyImageDataset
matplotlib.use('Qt5Agg')

# 2. Intializing the hyperparamters
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print(torch.__version__, DEVICE)
BATCH_SIZE = 100
EPOCHS = 10

# 3. Loading the puppy image dataset
# os.getcwd()
train_dataset = PuppyImageDataset(annotations_file='.\\data\\puppy\\annotation_input.csv',
                                  img_dir='.\\data\\puppy\\train',
                                  devide_by=5,
                                  remainder=[0, 1, 2, 3],
                                  transform=transforms.Resize([256, 256]))
valid_dataset = PuppyImageDataset(annotations_file='.\\data\\puppy\\annotation_input.csv',
                                  img_dir='.\\data\\puppy\\train',
                                  devide_by=5,
                                  remainder=[4],
                                  transform=transforms.Resize([256, 256]))
# test_dataset = PuppyImageDataset(annotations_file='.\\data\\puppy\\annotation_input_test.csv',
#                                  img_dir='.\\data\\puppy\\test',
#                                  transform=transforms.Resize([256, 256]))

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(dataset=valid_dataset, batch_size=BATCH_SIZE, shuffle=True)
# test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)





# check_img_load = next(iter(train_loader))[0]
# plt.imshow(check_img_load[2, :, :, :].permute(1, 2, 0))


# test_index = 12
# sample_image, sample_label = train_dataset.__getitem__(test_index)
# breed_index = train_dataset.img_labels.iloc[test_index, 1]
# breed_text = pd.read_csv('breed_matching_table.csv').iloc[breed_index, 0]
# plt.imshow(sample_image.permute(1, 2, 0))
# plt.title(breed_text)



