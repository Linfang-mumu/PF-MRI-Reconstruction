import numpy as np
import pandas as pd
import datetime
import torch.optim as optim
from scipy import io
import argparse
import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import h5py  
import matplotlib.pyplot as plt
import matplotlib
# print(matplotlib.__version__)
matplotlib.use('Agg')
import h5py  

from PIL import Image
import math
from sklearn.metrics import confusion_matrix
import pylab as pl
import numpy as np
import itertools
import torchvision
# from ._conv import register_converters as _register_converters

current_file_data_save = '//media/bisp/New Volume/Linfang/PF_CC398_218_170_218/PF55/SS/CC_brain_2D'
data_file = '/nfs/bisp_data_server/Linfang/CC359_dataset/multi_channel/train_val_12_channel/CC398_218_170_128/'
os.makedirs(current_file_data_save+'/test/', exist_ok=True)
os.makedirs(current_file_data_save+'/validation/', exist_ok=True)
os.makedirs(current_file_data_save+'/train/', exist_ok=True)
matrix_size = int(218)

pf_line = int(np.floor(matrix_size*0.45))
pf_line_com = matrix_size-pf_line
pf_line_o =int(np.floor(170*0.45)) 
SS_flag =1
MS_flag =0
EMS_flag =0
# for test
filePath =data_file +'/test/'
filename = os.listdir(filePath)
length = len(filename)
print('test')
for idx in range(0,length):
    f = h5py.File(filePath+ filename[idx],'r')
    # print(f.keys())
    # data = f['kspace'].value
    
    data_I = f['kspace_I'].value
    data_R = f['kspace_R'].value

    data = data_I*1j +data_R
    sz = data.shape

    freq= np.fft.ifftshift(data ,axes=0)
    freq = np.fft.ifft(freq,axis =0)
    data1 = np.sqrt(sz[0])*np.fft.fftshift(freq,axes=0)

    # data1 = data[(sz[0]//2-8):(sz[0]//2+8),(sz[1]//2-matix_crop):(sz[1]//2+matix_crop),(sz[2]//2-matix_crop):(sz[2]//2+matix_crop)]
    data2 = np.reshape(data1,(-1,1,sz[1],sz[2]))
    # sz1 = data1.shape
    # plt.figure(1)
    # plt.imshow(np.log(abs(data2[150,0, :, :])-1e-10), cmap='gray')
    # plt.savefig('k_or') 
    # # freq= np.fft.ifft2(data2 ,axes=(2,3))
    # # img = np.fft.ifftshift(freq,axes=(2,3))
    freq= np.fft.ifftshift(data2 ,axes=(2,3))
    freq = np.fft.ifft2(freq,axes=(2,3))
    img = np.sqrt(sz[1]*sz[2])*np.fft.fftshift(freq,axes=(2,3))
####################################################
    # plt.figure(1)
    # plt.imshow((abs(img[90,0, :, :])-1e-10), cmap='gray')
    # plt.savefig('img_data_or') 

    # plt.figure(1)
    # plt.imshow((np.angle(img[90,0, :, :])-1e-10), cmap='gray')
    # plt.savefig('img_angle_or')     
    # freq= np.fft.ifftshift(img ,axes=(2,3))
    # freq= np.fft.fft2(freq ,axes=(2,3))
    # freq = 1/matrix_size*np.fft.fftshift(freq,axes=(2,3))    
    # plt.figure(1)
    # plt.imshow(np.log(abs(freq[2,0, :, :])-1e-10), cmap='gray')
    # plt.savefig('k_or')
##############################################################
    scale_ref = round(np.max(np.absolute(img)),15)
    img = img/scale_ref
    ##single slice
    if SS_flag ==1:
        img_ref  = img[1:sz[0]-1,:,:,:]*0
        for k in range(1,sz[0]-1):
            img_ref[k-1,:,:,:] = img[k,:,:,:]
    else:
        img_ref = np.concatenate((img[1:sz[0]-1,:,:,:]*0,img[1:sz[0]-1,:,:,:]*0,img[1:sz[0]-1,:,:,:]*0),axis = 1)
        for k in range(1,sz[0]-1):
            img_ref[k-1,:,:,:] = np.concatenate((img[k,:,:,:],  img[k-1,:,:,:], img[k+1,:,:,:]), axis =0)         
    ##multi slice
    freq= np.fft.ifftshift(img_ref ,axes=(2,3))
    freq= np.fft.fft2(freq ,axes=(2,3))
    test_k= 1/np.sqrt(sz[1]*sz[2])*np.fft.fftshift(freq,axes=(2,3))   
    test_data = np.copy(test_k)
    test_data[:,0,:,:pf_line] = 0
    test_data[:,0,:pf_line_o,:] = 0    
##############complementry sampling pattern
    if EMS_flag ==1:
        test_data[:,1:3,:,pf_line_com:] = 0
    
    if MS_flag==1:
       test_data[:,1:3,:,:pf_line] = 0 
########################### 
    
    freq = np.fft.ifftshift(test_data,axes=(2,3))
    freq = np.fft.ifft2(freq ,axes=(2,3))
    img_data = np.sqrt(sz[1]*sz[2])*np.fft.fftshift(freq,axes=(2,3))   

    img_label =img_ref[:,0,:,:] -img_data[:,0,:,:]
    img_label =img_label.reshape(sz[0]-2,-1,sz[1],sz[2])

    imgdataR = np.copy(np.real(img_data))
    imgdataI = np.copy(np.imag(img_data))

    imgdata = np.concatenate((imgdataR,imgdataI), axis=1)

    imglabelR = np.copy(np.real(img_label))
    imglabelI = np.copy(np.imag(img_label))
    imglabel = np.concatenate((imglabelR,imglabelI), axis=1)

    imgfull=img_label[:,0,:,:] + img_data[:,0,:,:]
    imgfull =imgfull.reshape(sz[0]-2,-1,sz[1],sz[2])
   
#########################################################
    k_nor = np.fft.ifftshift(imgfull,axes=(2,3))  
    k_nor= np.fft.fft2(k_nor ,axes=(2,3))
    k_nor = 1/np.sqrt(sz[1]*sz[2])*np.fft.fftshift(k_nor,axes=(2,3))  

    k_zero = np.fft.ifftshift(img_data,axes=(2,3))  
    k_zero= np.fft.fft2(k_zero ,axes=(2,3)) 
    k_zero = 1/np.sqrt(sz[1]*sz[2])*np.fft.fftshift(k_zero,axes=(2,3))


    k_label = np.fft.ifftshift(img_label,axes=(2,3))  
    k_label= np.fft.fft2(k_label ,axes=(2,3))
    k_label = 1/np.sqrt(sz[1]*sz[2])*np.fft.fftshift(k_label,axes=(2,3))   

    nc =0
    nslice = 90
    plt.figure(1)    
    plt.imshow(np.log(np.abs(k_nor[nslice,0,:,:])),'gray')
    plt.savefig('0_original_k.png')

    plt.figure(2)
    plt.imshow(abs(imgfull[nslice,0,:,:]),'gray')
    plt.savefig('1_original_ref.png')

    plt.figure(3)    
    plt.imshow(np.log(np.abs(k_zero[nslice,nc,:,:])),'gray')
    plt.savefig('2_zero_k.png')

    plt.figure(4)
    plt.imshow(abs(img_data[nslice,nc,:,:]),'gray')
    plt.savefig('3_zero_img.png')

    plt.figure(6)
    plt.imshow(np.log(abs(k_label[nslice,0,:,:])),'gray')
    plt.savefig('4_label_k.png')

    plt.figure(5)
    plt.imshow(abs(img_label[nslice,0,:,:]),'gray')
    plt.savefig('5_label_img.png')
# ####################################################################   
#     plt.figure(3)    
#     plt.imshow(np.log(np.abs(k_zero[nslice,1,:,:])),'gray')
#     plt.savefig('2_zero_k_1.png')

#     plt.figure(4)
#     plt.imshow(abs(img_data[nslice,1,:,:]),'gray')
#     plt.savefig('3_zero_img_1.png')

#     plt.figure(3)    
#     plt.imshow(np.log(np.abs(k_zero[nslice,2,:,:])),'gray')
#     plt.savefig('2_zero_k_2.png')

#     plt.figure(4)
#     plt.imshow(abs(img_data[nslice,2,:,:]),'gray')
#     plt.savefig('3_zero_img_2.png')
#####################################################################   

    D = torch.from_numpy(imgdata).float()
    L = torch.from_numpy(imglabel).float()
    data = {'k-space':D,'label':L}
    torch.save(data,current_file_data_save +'/test/'+str(idx)+'.pth')
    f.close()
    
    
    
# # for validation
print('validation')
filePath =data_file +'/validation/'
filename = os.listdir(filePath)
length = len(filename)

for idx in range(0,length):
    f = h5py.File(filePath+ filename[idx],'r')

    data_I = f['kspace_I'].value
    data_R = f['kspace_R'].value

    data = data_I*1j +data_R
    sz = data.shape

    freq= np.fft.ifftshift(data ,axes=0)
    freq = np.fft.ifft(freq,axis =0)
    data1 = np.sqrt(sz[0])*np.fft.fftshift(freq,axes=0)

    data2 = np.reshape(data1,(-1,1,sz[1],sz[2]))

    freq= np.fft.ifftshift(data2 ,axes=(2,3))
    freq = np.fft.ifft2(freq,axes=(2,3))
    img = np.sqrt(sz[1]*sz[2])*np.fft.fftshift(freq,axes=(2,3))

    scale_ref = round(np.max(np.absolute(img)),15)
    img = img/scale_ref

    ##single slice
    if SS_flag ==1:
        img_ref  = img[1:sz[0]-1,:,:,:]*0
        for k in range(1,sz[0]-1):
            img_ref[k-1,:,:,:] = img[k,:,:,:]
    else:
        img_ref = np.concatenate((img[1:sz[0]-1,:,:,:]*0,img[1:sz[0]-1,:,:,:]*0,img[1:sz[0]-1,:,:,:]*0),axis = 1)
        for k in range(1,sz[0]-1):
            img_ref[k-1,:,:,:] = np.concatenate((img[k,:,:,:],  img[k-1,:,:,:], img[k+1,:,:,:]), axis =0)          

    ##multi slice
    freq= np.fft.ifftshift(img_ref ,axes=(2,3))
    freq= np.fft.fft2(freq ,axes=(2,3))
    test_k= 1/np.sqrt(sz[1]*sz[2])*np.fft.fftshift(freq,axes=(2,3))   
    test_data = np.copy(test_k)
    test_data[:,0,:,:pf_line] = 0

    test_data[:,0,:pf_line_o,:] = 0    
##############complementry sampling pattern
    if EMS_flag ==1:
        test_data[:,1:3,:,pf_line_com:] = 0
    
    if MS_flag==1:
       test_data[:,1:3,:,:pf_line] = 0 
########################### 
    
    freq = np.fft.ifftshift(test_data,axes=(2,3))
    freq = np.fft.ifft2(freq ,axes=(2,3))
    img_data = np.sqrt(sz[1]*sz[2])*np.fft.fftshift(freq,axes=(2,3))   

    img_label =img_ref[:,0,:,:] -img_data[:,0,:,:]
    img_label =img_label.reshape(sz[0]-2,-1,sz[1],sz[2])

    imgdataR = np.copy(np.real(img_data))
    imgdataI = np.copy(np.imag(img_data))

    imgdata = np.concatenate((imgdataR,imgdataI), axis=1)



    imglabelR = np.copy(np.real(img_label))
    imglabelI = np.copy(np.imag(img_label))
    imglabel = np.concatenate((imglabelR,imglabelI), axis=1)


    imgfull=img_label[:,0,:,:] + img_data[:,0,:,:]
    imgfull =imgfull.reshape(sz[0]-2,-1,sz[1],sz[2])
 
    for i in range(16,int(sz[0]-20)):
        D = torch.from_numpy(imgdata[i,:,:,:]).float()
        L = torch.from_numpy(imglabel[i,:,:,:]).float()
        data = {'k-space':D,'label':L}
        torch.save(data,current_file_data_save +'/validation/'+str(idx)+'_'+str(i)+'.pth')
    f.close()
    
    
# # for train
print('train')
filePath =data_file +'/train/'
filename = os.listdir(filePath)
length = len(filename)

for idx in range(0,length):
    f = h5py.File(filePath+ filename[idx],'r')
    data_I = f['kspace_I'].value
    data_R = f['kspace_R'].value

    data = data_I*1j +data_R
    sz = data.shape

    freq= np.fft.ifftshift(data ,axes=0)
    freq = np.fft.ifft(freq,axis =0)
    data1 = np.sqrt(sz[0])*np.fft.fftshift(freq,axes=0)
    data2 = np.reshape(data1,(-1,1,sz[1],sz[2]))

    freq= np.fft.ifftshift(data2 ,axes=(2,3))
    freq = np.fft.ifft2(freq,axes=(2,3))
    img = np.sqrt(sz[1]*sz[2])*np.fft.fftshift(freq,axes=(2,3))

    scale_ref = round(np.max(np.absolute(img)),15)
    img = img/scale_ref





    ##single slice
    if SS_flag ==1:
        img_ref  = img[1:sz[0]-1,:,:,:]*0
        for k in range(1,sz[0]-1):
            img_ref[k-1,:,:,:] = img[k,:,:,:]
    else:
        img_ref = np.concatenate((img[1:sz[0]-1,:,:,:]*0,img[1:sz[0]-1,:,:,:]*0,img[1:sz[0]-1,:,:,:]*0),axis = 1)
        for k in range(1,sz[0]-1):
            img_ref[k-1,:,:,:] = np.concatenate((img[k,:,:,:],  img[k-1,:,:,:], img[k+1,:,:,:]), axis =0)          

    ##multi slice
    freq= np.fft.ifftshift(img_ref ,axes=(2,3))
    freq= np.fft.fft2(freq ,axes=(2,3))
    test_k= 1/np.sqrt(sz[1]*sz[2])*np.fft.fftshift(freq,axes=(2,3))   
    test_data = np.copy(test_k)
    test_data[:,0,:,:pf_line] = 0

    test_data[:,0,:pf_line_o,:] = 0    
##############complementry sampling pattern
    if EMS_flag ==1:
        test_data[:,1:3,:,pf_line_com:] = 0
    
    if MS_flag==1:
       test_data[:,1:3,:,:pf_line] = 0 
########################### 
    
    freq = np.fft.ifftshift(test_data,axes=(2,3))
    freq = np.fft.ifft2(freq ,axes=(2,3))
    img_data = np.sqrt(sz[1]*sz[2])*np.fft.fftshift(freq,axes=(2,3))   

    img_label =img_ref[:,0,:,:] -img_data[:,0,:,:]
    img_label =img_label.reshape(sz[0]-2,-1,sz[1],sz[2])

    imgdataR = np.copy(np.real(img_data))
    imgdataI = np.copy(np.imag(img_data))

    imgdata = np.concatenate((imgdataR,imgdataI), axis=1)



    imglabelR = np.copy(np.real(img_label))
    imglabelI = np.copy(np.imag(img_label))
    imglabel = np.concatenate((imglabelR,imglabelI), axis=1)


    imgfull=img_label[:,0,:,:] + img_data[:,0,:,:]
    imgfull =imgfull.reshape(sz[0]-2,-1,sz[1],sz[2])
 
    for i in range(16,int(sz[0]-20)):
        D = torch.from_numpy(imgdata[i,:,:,:]).float()
        L = torch.from_numpy(imglabel[i,:,:,:]).float()
        data = {'k-space':D,'label':L}
        torch.save(data,current_file_data_save +'/train/'+str(idx)+'_'+str(i)+'.pth')
    f.close()

os.system('python CNN_train_2d.py')