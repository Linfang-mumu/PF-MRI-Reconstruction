import numpy as np
import pandas as pd
import datetime
# from model2 import Net1, Net2
from model2_real_2D import Net_real
# from model2_cpx import Net_cpx
import torch.optim as optim
from scipy import io
import argparse
import os
import torch
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
import math
from losses import SSIMLoss2D_MC
import time

os.environ["CUDA_VISIBLE_DEVICES"]="0" #USE gpu 1, gp0 cannot be used for some reason
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
num_workers = 0

current_file = '//media/bisp/New Volume/Linfang/PF_CC398_218_170_218/PF60/SS//'
# current_file ='//media/bisp/New Volume/Linfang/PF_DL_paper_FSE/PF55/SS/'
# current_file_name = current_file + 'test_NYU_brain_0316_new/'
# current_file_name = current_file + 'NYU_brain_T1_complex/'
current_file_name = current_file + '/CC_brain_2D/'
# current_file_name = current_file + '/NYU_knee/'


class prepareData(Dataset):
    def __init__(self, train_or_test):

       self.files = os.listdir(current_file_name+train_or_test)
       self.train_or_test= train_or_test

    def __len__(self):
       return len(self.files)

    def __getitem__(self, idx):
        
        data = torch.load(current_file_name+self.train_or_test+'/'+self.files[idx])
        return data['k-space'],  data['label']


testset = prepareData('test')

# testset = prepareData('test_'+flag)
testloader = torch.utils.data.DataLoader(testset, batch_size=1,shuffle=False, num_workers=num_workers) 
# model = torch.load(current_file+'/real_64_16'+'/epoch-33-0.0128679229.pth')# ss
# model = torch.load(current_file+'/cpx_64_16'+'/epoch-33-0.0146769155.pth')# ss 
# model = torch.load('/media/bisp/New Volume/Linfang/PF_DL_paper_cpx_ssim/PF55/SS/real_ssim_64_16'+'/epoch-9-0.0866945386.pth')# repeat once  
# model = torch.load(current_file+'/real_ssim_64_16_swi'+'/epoch-10-0.0677850842.pth')# ss  
# model = torch.load(current_file+'/real_ssim_64_16'+'/epoch-16-0.0836557150.pth')# ss60 
# model = torch.load(current_file+'/real_ssim_64_16'+'/epoch-18-0.0791845322.pth')# ss55
model = torch.load(current_file+'/real_L1_64_16_cpx'+'/epoch-15-0.0184234809.pth')# ss51  
nx=218
ny=170  
nc =2      
print(model)
save_file = '/real_64_16_ssim_brain' +'/'
# save_file = '/real_64_16'  +'/'
criterion1 = nn.L1Loss()
ssim = SSIMLoss2D_MC(in_chan=2)
model.eval()
loss_validation_list = []
loss_batch = []
loss = []
print('\n testing...')
time_start=time.time() 
for i, data in enumerate(testloader, 0):
    inputs = data[0].reshape(-1,nc,ny,nx).to(device)
    label = data[1].reshape(-1,2,ny,nx).to(device)
    labels = inputs +label

    os.makedirs(current_file_name+save_file+str(i), exist_ok=True)    
    inpin = torch.complex(inputs[:,0,:,:],inputs[:,1,:,:])
    inpin = inpin.cpu()
    sz = inpin.size()
    nslice = sz[0]//2
    plt.figure(2)
    plt.imshow(abs(inpin[nslice,:,:]),cmap='gray')
    plt.savefig(current_file_name+save_file+str(i)+'/8_inpin')
    plt.figure(2)
    plt.imshow(np.angle(inpin[nslice,:,:]),cmap='gray')
    plt.savefig(current_file_name+save_file+str(i)+'/18_inpin_angle')

    ref = torch.complex(labels[:,0,:,:],labels[:,1,:,:])
    ref = ref.cpu()
    plt.figure(1)
    plt.imshow(np.abs(ref[nslice,:,:]),cmap='gray')
    plt.savefig(current_file_name+save_file+str(i)+'/8_ref')
    plt.figure(1)
    plt.imshow(np.angle(ref[nslice,:,:]),cmap='gray')
    plt.savefig(current_file_name+save_file+str(i)+'/18_ref_angle')
    

    labelin =torch.complex(labels[:,0,:,:],labels[:,1,:,:])-torch.complex(inputs[:,0,:,:],inputs[:,1,:,:])
    labelin = labelin.cpu()
    plt.figure(3)
    plt.imshow(np.abs(labelin[nslice,:,:]),cmap='gray')
    plt.savefig(current_file_name+save_file+str(i)+'/6_label')


    ref_ksp =  torch.fft.ifftshift(ref,dim=(1, 2))
    ref_k=  torch.fft.fft2(ref_ksp ,dim=(1, 2))
    ref_k =1/math.sqrt(sz[1]*sz[2])* torch.fft.fftshift(ref_k,dim=(1, 2))   
## for 3d k-sapce
    # ref_k= np.fft.fftshift(ref_k,axes=0)
    # ref_k = np.fft.fft(ref_k,axis=0) 

    plt.figure(4)
    plt.imshow(torch.log(abs(ref_k[nslice,:,:])),cmap='gray')
    plt.savefig(current_file_name+save_file+str(i)+'/7_ref_k') 
    
    # img to k-space
    inp_ksp =  torch.fft.ifftshift(inpin,dim=(1, 2))
    inp_k= torch.fft.fft2(inp_ksp ,dim=(1, 2))
    inp_k =  1/math.sqrt(sz[1]*sz[2])*torch.fft.fftshift(inp_k,dim=(1, 2))
## for 3d k-sapce
    # inp_k= np.fft.fftshift(inp_k,axes=0)
    # inp_k = np.fft.fft(inp_k,axis=0) 

    plt.figure(5)
    plt.imshow(torch.log(torch.abs(inp_k[nslice,:,:])),cmap='gray')
    plt.savefig(current_file_name+save_file+str(i)+'/7_inp_k') 
    
    # print(label.shape)
    # plt.imshow(np.log(np.abs(label[1,:,:])),cmap='gray')
    # freq= np.fft.ifft2(label,axes=(2,3))
    # img= np.fft.ifftshift(freq,axes=(2,3))
    # plt.figure(1)
    # plt.imshow(np.abs(img[12,0,:,:]),cmap='gray')
    # plt.show()
   
#     Lable_tumor.append(labels)
    with torch.no_grad():
         outs = model(inputs)





    # la_out = outs.cuda().data.cpu() 

    # la_out= la_out.numpy() 
        
    output = torch.complex(outs[:,0,:,:],outs[:,1,:,:])
    output = output.cpu()
    plt.figure(6)
    plt.imshow(abs(output[nslice,:,:]),cmap='gray')
    plt.savefig(current_file_name+save_file+str(i)+'/8_output_ref')
    plt.figure(6)
    plt.imshow(np.angle(output[nslice,:,:]),cmap='gray')
    plt.savefig(current_file_name+save_file+str(i)+'/18_output_ref_angle')    

    label_out =torch.complex(outs[:,0,:,:],outs[:,1,:,:])-torch.complex(inputs[:,0,:,:],inputs[:,1,:,:])
    label_out = label_out.cpu()
    plt.figure(7)
    plt.imshow(abs(label_out[nslice,:,:]),cmap='gray')
    plt.savefig(current_file_name+save_file+str(i)+'/6_output_label')
    
    # img to k-space
    outp_ksp =  torch.fft.ifftshift(output,dim=(1,2))
    outp_k= torch.fft.fft2(outp_ksp ,dim=(1,2))
    outp_k= 1/math.sqrt(sz[1]*sz[2])*torch.fft.fftshift(outp_k,dim=(1,2))
## for 3d k-sapce
    # outp_k= np.fft.fftshift(outp_k,axes=0)
    # outp_k = np.fft.fft(outp_k,axis=0) 

    plt.figure(8)
    plt.imshow(torch.log(abs(outp_k[nslice,:,:])),cmap='gray')
    plt.savefig(current_file_name+save_file+str(i)+'/7_outp_k') 

    plt.figure(9)
    residual = output-ref
    plt.imshow(abs(residual[nslice,:,:]),cmap='gray')
    plt.savefig(current_file_name+save_file+str(i)+'/9_residual')     
    
    # loss = criterion1(outs, labels)
    loss = ssim(outs, labels,1)   
    loss_batch.append(loss.item())
    loss_validation_list.append(round(sum(loss_batch) / len(loss_batch),10))
    print(loss_validation_list)
    # output = outs.cuda().data.cpu()
    # labelo =labels.cuda().data.cpu()
    # inputo = inputs.cuda().data.cpu()
    # outp_k = outp_k[1:-1,:,:]
    # ref_k = ref_k[1:-1,:,:]
    # inp_k = inp_k[1:-1,:,:]
    os.makedirs(current_file_name+save_file+'/outputs/', exist_ok=True)
    os.makedirs(current_file_name+save_file+'/reference/', exist_ok=True)
    os.makedirs(current_file_name+save_file+'/inputs/', exist_ok=True)
    f = h5py.File(current_file_name+save_file+'/outputs/'+str(i)+'.h5','w')
    f['k-space'] = outp_k
    g = h5py.File(current_file_name+save_file+'/reference/'+str(i)+'.h5','w')
    g['k-space'] = ref_k
    k = h5py.File(current_file_name+save_file+'/inputs/'+str(i)+'.h5','w')
    k['k-space'] = inp_k
    # f = h5py.File(current_file_name+save_file+'/outputs/'+str(i)+'.h5','w')
    # f['k-space'] = output
    # g = h5py.File(current_file_name+save_file+'/reference/'+str(i)+'.h5','w')
    # g['k-space'] = ref
    # k = h5py.File(current_file_name+save_file+'/inputs/'+str(i)+'.h5','w')
    # k['k-space'] = inpin
    f.close()
    g.close()
    k.close()
    time_end=time.time()
    print('time cost for testing',time_end-time_start,'s')   
        