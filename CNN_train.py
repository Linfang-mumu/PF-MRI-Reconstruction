#------------------------- python ------------------------------------------
# jupyter notebook
import numpy as np
import pandas as pd
import datetime
from model2_cpx import Net_cpx, Net_MS_cpx
import time

import torch.optim as optim
from scipy import io
import argparse
import os                    # nn.BatchNorm2d(2,affine=False),
import torch
# from SSIM import SSIM
from losses import SSIMLoss2D_MC

from torch import nn
from torch.utils.data import Dataset, DataLoader
import h5py  
import matplotlib.pyplot as plt
import h5py  
import matplotlib
from PIL import Image
import math
from sklearn.metrics import confusion_matrix
import pylab as pl
import matplotlib.pyplot as plt
import numpy as np
import itertools

os.environ["CUDA_VISIBLE_DEVICES"]="0" #USE gpu 1, gp0 cannot be used for some reason
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print(device)
epoch_num = 50 #itration number 
num_workers = 0

current_data= '//media/bisp/New Volume/Linfang/PF_CC398_218_170_218/PF55/MS/'
current_data_file = current_data + 'CC_brain/'

os.makedirs(current_data+'/ssim_64_16_cpx'+'/', exist_ok=True)
model_save_path = current_data + '/ssim_64_16_cpx'+'/'
class prepareData(Dataset):
    def __init__(self, train_or_test):

       self.files = os.listdir(current_data_file+train_or_test)
       self.train_or_test= train_or_test

    def __len__(self):
       return len(self.files)

    def __getitem__(self, idx):
        c=current_data_file+self.train_or_test+'/'+self.files[idx]
        
        data = torch.load(current_data_file+self.train_or_test+'/'+self.files[idx])
        return data['k-space'],  data['label']
  
trainset = prepareData('train')
trainloader = torch.utils.data.DataLoader(trainset, batch_size=16,shuffle=True, num_workers=num_workers)

validationset = prepareData('validation')
validationloader = torch.utils.data.DataLoader(validationset, batch_size=1,shuffle=True, num_workers=num_workers)

testset = prepareData('test')
testloader = torch.utils.data.DataLoader(testset, batch_size=1,shuffle=False, num_workers=num_workers)

# model = Net_cpx().to(device)
model = Net_MS_cpx().to(device)
print(model)

criterion1 = nn.L1Loss()

lr = 0.0002
nx = 218
ny = 170
# nx = 256
# ny = 256
nc = 6
weight_decay = 0.000
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

ssim = SSIMLoss2D_MC(in_chan=2)
loss_train_list = []
loss_validation_list = []

for epoch in range(epoch_num):   #set to 0 for no running the training
    model.train()

    loss_batch = []
    time_start=time.time()    
    for i, data in enumerate(trainloader, 0):
        inputs = data[0].reshape(-1,nc,ny,nx).to(device) ##single slice
        label = data[1].reshape(-1,2,ny,nx).to(device) 
        if nc == 6:
            labels= label
            labels[:,0,:,:]= label[:,0,:,:] +inputs[:,0,:,:]
            labels[:,1,:,:]= label[:,1,:,:] +inputs[:,3,:,:]
        else:
            labels = inputs + label    

        outs = model(inputs)
        loss = criterion1(outs, labels)
        # loss = ssim(outs, labels,1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_batch.append(loss.item())  
        if (i)%10==0:
            print('epoch:%d - %d, loss:%.10f'%(epoch+1,i+1,loss.item()))
        # break
        # h=0
    loss_train_list.append(round(sum(loss_batch) / len(loss_batch),10))
    print(loss_train_list)
    time_end=time.time()
    print('time cost for training',time_end-time_start,'s')
    model.eval()     # evaluation
    loss_batch = []
    print('\n testing...')
    time_start=time.time()
    for i, data in enumerate(validationloader, 0):
        inputs = data[0].reshape(-1,nc,ny,nx).to(device)
        label = data[1].reshape(-1,2,ny,nx).to(device) 
        if nc ==6:
            labels= label
            labels[:,0,:,:]= label[:,0,:,:] +inputs[:,0,:,:]
            labels[:,1,:,:]= label[:,1,:,:] +inputs[:,3,:,:]
        else:
            labels = inputs + label

        with torch.no_grad():
            outs = model(inputs)
        loss = criterion1(outs, labels)
        # loss = ssim(outs, labels,1)    ####using the L1loss for several epochs, then using ssim to train the whole model
        loss_batch.append(loss.item())
        
    time_end=time.time()
    print('time cost for testing',time_end-time_start,'s')
    loss_validation_list.append(round(sum(loss_batch) / len(loss_batch),10))
    print(loss_validation_list)

    torch.save(model, os.path.join(model_save_path, 'epoch-%d-%.10f.pth' % (epoch+1, loss.item())))

    # if (epoch+1) % 2 == 0:
    #     lr = max(5e-5,lr*0.8)
    #     optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

print('Finished Training')