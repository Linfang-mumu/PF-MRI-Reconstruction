import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torchvision

RB =0
class ComplexConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=True):
        super(ComplexConv2d, self).__init__()
        padding = kernel_size // 2
        self.conv_r = nn.Conv2d(in_channels, out_channels,
                                kernel_size, stride, padding, bias)
        self.conv_i = nn.Conv2d(in_channels, out_channels,
                                kernel_size, stride, padding, bias)

    def forward(self, x):

        input_r, input_i = torch.split(x, x.shape[1]//2, dim=1)

        y1 = self.conv_r(input_r)-self.conv_i(input_i)
        y2 = self.conv_r(input_i)+self.conv_i(input_r)

        return torch.cat((y1, y2), dim=1)

class MeanShift(nn.Conv2d):
    def __init__(
            self, rgb_range,
            rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):

        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


class Upsampler2(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act='relu', bias=True):

        m = []
        for _ in range(int(math.log(scale, 2))):
            m.append(nn.UpsamplingNearest2d(scale_factor=2))
            m.append(conv(n_feats, n_feats, 3, bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))

        super(Upsampler2, self).__init__(*m)


class BasicBlock(nn.Sequential):
    def __init__(
            self, conv, in_channels, out_channels, kernel_size, stride=1, bias=False,
            bn=True, act=nn.ReLU(True)):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)

class ResBlock(nn.Module):
    def __init__(
        self, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(ComplexConv2d(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res

class Net_cpx(nn.Module):
    def __init__(self):
        super(Net_cpx, self).__init__()

        n_resblocks = 16
        n_feats = 32

        kernel_size = 3
        self.scale = 1
        n_colors = 1

        # act = ComplexReLU()
        # act = nn.ReLU(True)
        # self.sub_mean = MeanShift(rgb_range)
        # self.add_mean = MeanShift(rgb_range, sign=1)

        # define head module
        # m_head = ComplexConv2d(n_colors, n_feats, kernel_size)

        # define body module
        m_body = [
            ResBlock(n_feats, kernel_size,  res_scale=1
                     ) for _ in range(n_resblocks)
        ]
        m_body.append(ComplexConv2d(n_feats, n_feats, kernel_size))

        # define tail module

        # m_tail = [
        #     ComplexConv2d(n_feats, n_colors, kernel_size)
        # ]

        self.head = ComplexConv2d(n_colors, n_feats, kernel_size)
        self.body = nn.Sequential(*m_body)
        self.tail = ComplexConv2d(n_feats, n_colors, kernel_size)

        # self.cc = ComplexConv2d(1,64)

        self.register_parameter("t", nn.Parameter(-2*torch.ones(1)))

    def forward(self, x):
        or_im = x
        nsize =x.size()
        pf = math.floor(nsize[3]*0.45)
        pf_com =nsize[3] -pf
        # print(nsize[3])
        # pf_ratio =torch.int(torch.floor(nsize[3]*0.45))
        or_k = torch.complex(or_im[:, 0, :, :], or_im[:, 1, :, :])
        or_k = torch.fft.ifftshift(or_k, dim=(1, 2))
        or_k = torch.fft.fft2(or_k, dim=(1, 2))
        or_k = 1/math.sqrt(nsize[2]*nsize[3])*torch.fft.fftshift(or_k, dim=(1, 2))

######################################################
        # or_k_ = or_k
        # or_k_ = or_k_.cpu()
        # plt.figure(6)
        # plt.imshow(np.log(abs(or_k_.detach_().numpy()[2, :, :])-1e-10), cmap='gray')
        # plt.savefig('input_k_or')

        # or_im_ = torch.complex(or_im[:,0,:,:],or_im[:,1,:,:])
        # or_im_ = or_im_.cpu()
        # plt.figure(7)
        # plt.imshow((abs(or_im_.detach_().numpy()[2, :, :])), cmap='gray')
        # plt.savefig('input_img_or')
#################################################

        y = x
        for i in range(2):
            x = y
            new_k = torch.complex(x[:, 0, :, :], x[:, 1, :, :])

###################################            
            # new_k_ = new_k
            # new_k_ = new_k_.cpu()
            # plt.figure(70)
            # plt.imshow((abs(new_k_.detach_().numpy()[2, :, :])), cmap='gray')
            # plt.savefig('iter_'+str(i)+'_img_before_DC')
################################### 
            new_k = torch.fft.ifftshift(new_k, dim=(1,2))
            new_k = torch.fft.fft2(new_k, dim=(1, 2))
            new_k = 1/math.sqrt(nsize[2]*nsize[3])*torch.fft.fftshift(new_k, dim=(1,2))

################################### 
            # new_k_ = new_k
            # new_k_ = new_k_.cpu()
            # plt.figure(80)
            # plt.imshow(np.log(abs(new_k_.detach_().numpy()[2, :, :])-1e-10), cmap='gray')
            # plt.savefig('iter_'+str(i)+'_kspace_before_DC')

################################### 
            new_k[:, :, :pf] = or_k[:, :, :pf]  # only keep the measured data
            new_k = torch.fft.ifftshift(new_k, dim=(1, 2))            
            new_k = torch.fft.ifft2(new_k, dim=(1, 2))
            new_k = math.sqrt(nsize[2]*nsize[3])*torch.fft.fftshift(new_k, dim=(1, 2))
            new_k = torch.stack((torch.real(new_k), torch.imag(new_k)), dim=1)            
            x = x + self.t*(new_k - or_im) # t learnable parameter  ## only keep the measured data

# ################################### 
            # test_im = torch.complex(x[:, 0, :, :], x[:, 1, :, :])
            # # test_k = test_im
            # test_k = torch.fft.ifftshift(test_im, dim=(1, 2))
            # test_k = torch.fft.fft2(test_k, dim=(1, 2))
            # test_k = 1/math.sqrt(nsize[2]*nsize[3])*torch.fft.fftshift(test_k, dim=(1, 2))

            # test_k_ = test_k
            # test_k_ = test_k_.cpu()
            # plt.figure(10)
            # plt.imshow(np.log(abs(test_k_.detach_().numpy()[2, :, :])-1e-10), cmap='gray')
            # plt.savefig('iter_'+str(i)+'_kspace_after_soft_DC')

            # x_ = torch.complex(x[:, 0, :, :], x[:, 1, :, :])
            # x_ = x_.cpu()
            # plt.figure(61)
            # plt.imshow(abs(x_.detach_().numpy()[2, :, :]), cmap='gray')
            # plt.savefig('iter_'+str(i)+'_img_after_soft_DC_input')
            
###################################
#  
            res = self.head(x)
###################################            
            # test_im = res
            # test_im = torch.complex(res[:, :64, :, :], res[:, 64:, :, :])
            # # test_k = test_im
            # test_k = torch.fft.fftshift(test_im, dim=(2, 3))
            # test_k = torch.fft.fft2(test_k, dim=(2, 3))
            # test_k = 1/math.sqrt(nsize[2]*nsize[3])*torch.fft.fftshift(test_k, dim=(2, 3))
            # res_dis = test_im.permute(1,0,2,3)
            # slice1 = torch.reshape(res_dis[:,2,:,:],(64,1,256,256))
            # torchvision.utils.save_image(abs(slice1),'iter_'+str(i)+'_img_head_output.png',normalize=True)

            # res_dis = test_k.permute(1,0,2,3)
            # slice1 = torch.reshape(res_dis[:,2,:,:],(64,1,256,256))
            # torchvision.utils.save_image(torch.log(abs(slice1)),'iter_'+str(i)+'_k_head_output.png',normalize=True)
###################################  




            res = self.body(res)

# ###################################              
            # test_im = torch.complex(res[:, :64, :, :], res[:, 64:, :, :])
            # # test_k = test_im
            # test_k = torch.fft.ifftshift(test_im, dim=(2, 3))
            # test_k = torch.fft.fft2(test_k, dim=(2, 3))
            # test_k = 1/math.sqrt(nsize[2]*nsize[3])*torch.fft.fftshift(test_k, dim=(2, 3))
            # res_dis = test_im.permute(1,0,2,3)
            # slice1 = torch.reshape(res_dis[:,2,:,:],(64,1,256,256))
            # torchvision.utils.save_image(abs(slice1),'iter_'+str(i)+'_img_body_output.png',normalize=True,scale_each = True)

            # res_dis = test_k.permute(1,0,2,3)
            # slice1 = torch.reshape(res_dis[:,2,:,:],(64,1,256,256))
            # torchvision.utils.save_image(torch.log(abs(slice1)),'iter_'+str(i)+'_k_body_output.png',normalize=True,scale_each = True)


            # y = self.tail(res)
            # test_im = torch.complex(y[:, 0, :, :], y[:, 1, :, :])
            # # test_k =test_im
            # test_k = torch.fft.fftshift(test_im, dim=(1, 2))
            # test_k = torch.fft.fft2(test_k, dim=(1, 2))
            # test_k = 1/math.sqrt(nsize[2]*nsize[3])*torch.fft.fftshift(test_k, dim=(1, 2))

            # test_k_ = test_k
            # test_k_ = test_k_.cpu()
            # plt.figure(10)
            # plt.imshow(
            # np.log(abs(test_k_.detach_().numpy()[2, :, :])-1e-10), cmap='gray')
            # plt.savefig('iter_'+str(i)+'_kspace_tail_output_residual')            


            # y_ = torch.complex(y[:, 0, :, :], y[:, 1, :, :])
            # y_ = y_.cpu()
            # plt.figure(71)
            # plt.imshow(abs(y_.detach_().numpy()[2, :, :]), cmap='gray')
            # plt.savefig('iter_'+str(i)+'_img_tail_output_residual')
###################################     
            y = self.tail(res) + x   
#####################################            

        #     test_im = torch.complex(y[:, 0, :, :], y[:, 1, :, :])
        #     test_k = test_im
        #     test_k = torch.fft.ifftshift(test_im, dim=(1, 2))
        #     test_k = torch.fft.fft2(test_k, dim=(1, 2))
        #     test_k = 1/math.sqrt(nsize[2]*nsize[3])*torch.fft.fftshift(test_k, dim=(1, 2))
        #     test_k_ = test_k
        #     test_k_ = test_k_.cpu()
        #     plt.figure(10)
        #     plt.imshow(np.log(abs(test_k_.detach_().numpy()[2, :, :])-1e-10), cmap='gray')
        #     plt.savefig('iter_'+str(i)+'_kspace_tail_output')


        #     y_ = torch.complex(y[:, 0, :, :], y[:, 1, :, :])
        #     y_ = y_.cpu()
        #     plt.figure(71)
        #     plt.imshow(abs(y_.detach_().numpy()[2, :, :]), cmap='gray')
        #     plt.savefig('iter_'+str(i)+'_img_tail_output')

        # plt.figure(62)
        # plt.imshow(abs(y_.detach_().numpy()[2, :, :]), cmap='gray')
        # plt.savefig('img_before_hard_DC')
###################################
        new_k = torch.complex(y[:, 0, :, :], y[:, 1, :, :])
        new_k = torch.fft.ifftshift(new_k, dim=(1, 2))
        new_k = torch.fft.fft2(new_k, dim=(1, 2))
        new_k = 1/math.sqrt(nsize[2]*nsize[3])*torch.fft.fftshift(new_k, dim=(1, 2))
###################################
        # new_k_ = new_k
        # new_k_ = new_k_.cpu()
        # plt.figure(62)
        # plt.imshow(np.log(abs(new_k_.detach_().numpy()[2, :, :])), cmap='gray')
        # plt.savefig('k_before_hard_DC')
###################################
        new_k[:, :, pf_com:] = or_k[:, :, pf_com:]
###################################
        # new_k_ = new_k
        # new_k_ = new_k_.cpu()
        # plt.figure(62)
        # plt.imshow(np.log(abs(new_k_.detach_().numpy()[2, :, :])), cmap='gray')
        # plt.savefig('k_after_hard_DC')
###################################
        new_k = torch.fft.ifftshift(new_k, dim=(1, 2))
        new_k = torch.fft.ifft2(new_k, dim=(1, 2))
        new_k = math.sqrt(nsize[2]*nsize[3])*torch.fft.fftshift(new_k, dim=(1, 2))
###################################
        # new_k_ = new_k
        # new_k_ = new_k_.cpu()
        # plt.figure(6)
        # plt.imshow(abs(new_k_.detach_().numpy()[2, :, :]), cmap='gray')
        # plt.savefig('img_after_hard_DC')
###################################
        y = torch.stack((torch.real(new_k), torch.imag(new_k)), dim=1)
        return y



class Net_MS_cpx(nn.Module):
    def __init__(self):
        super(Net_MS_cpx, self).__init__()

        n_resblocks = 16
        n_feats = 32

        kernel_size = 3
        self.scale = 1
        n_colors = 1

        # act = ComplexReLU()
        # act = nn.ReLU(True)
        # self.sub_mean = MeanShift(rgb_range)
        # self.add_mean = MeanShift(rgb_range, sign=1)

        # define head module
        # m_head = ComplexConv2d(n_colors, n_feats, kernel_size)

        # define body module
        m_body = [
            ResBlock(n_feats, kernel_size,  res_scale=1
                     ) for _ in range(n_resblocks)
        ]
        m_body.append(ComplexConv2d(n_feats, n_feats, kernel_size))

        # define tail module

        # m_tail = [
        #     ComplexConv2d(n_feats, n_colors, kernel_size)
        # ]

        self.head = ComplexConv2d(3, n_feats, kernel_size)
        self.body = nn.Sequential(*m_body)
        self.tail = ComplexConv2d(n_feats, n_colors, kernel_size)

        # self.cc = ComplexConv2d(1,64)

        self.register_parameter("t", nn.Parameter(-2*torch.ones(1)))

    def forward(self, x):
        # or_im = x
        nsize =x.size()
        pf = math.floor(nsize[3]*0.40)
        pf_com =nsize[3] -pf
        # print(nsize[3])
        # pf_ratio =torch.int(torch.floor(nsize[3]*0.45))
        or_im = torch.complex(x[:, 0, :, :], x[:, 3, :, :]) #central slice
        # nsize =or_im.size()
        or_k = torch.fft.ifftshift(or_im, dim=(1, 2))
        or_k = torch.fft.fft2(or_k, dim=(1, 2))
        or_k = 1/math.sqrt(nsize[2]*nsize[3])*torch.fft.fftshift(or_k, dim=(1, 2))

######################################################
        # or_k_ = or_k
        # or_k_ = or_k_.cpu()
        # plt.figure(6)
        # plt.imshow(np.log(abs(or_k_.detach_().numpy()[2, :, :])-1e-10), cmap='gray')
        # plt.savefig('input_k_or')

        # or_im_ = torch.complex(or_im[:,0,:,:],or_im[:,1,:,:])
        # or_im_ = or_im_.cpu()
        # plt.figure(7)
        # plt.imshow((abs(or_im_.detach_().numpy()[2, :, :])), cmap='gray')
        # plt.savefig('input_img_or')
#################################################
        

        x_or = x
        y = x[:,0:2,:,:]
        y[:,1,:,:] = x[:,3,:,:] #the central slice
        or_im = y

        for i in range(2):
            x = x_or
            x[:,0,:,:].data = y[:,0,:,:]
            x[:,3,:,:].data = y[:,1,:,:].data

            new_k = torch.complex(x[:, 0, :, :], x[:, 3, :, :])
###################################            
            # new_k_ = new_k
            # new_k_ = new_k_.cpu()
            # plt.figure(70)
            # plt.imshow((abs(new_k_.detach_().numpy()[2, :, :])), cmap='gray')
            # plt.savefig('iter_'+str(i)+'_img_before_DC')
################################### 
            new_k = torch.fft.ifftshift(new_k, dim=(1,2))
            new_k = torch.fft.fft2(new_k, dim=(1, 2))
            new_k = 1/math.sqrt(nsize[2]*nsize[3])*torch.fft.fftshift(new_k, dim=(1,2))

################################### 
            # new_k_ = new_k
            # new_k_ = new_k_.cpu()
            # plt.figure(80)
            # plt.imshow(np.log(abs(new_k_.detach_().numpy()[2, :, :])-1e-10), cmap='gray')
            # plt.savefig('iter_'+str(i)+'_kspace_before_DC')

################################### 
            new_k[:, :, :pf] = or_k[:, :, :pf]  # only keep the measured data
            new_k = torch.fft.ifftshift(new_k, dim=(1, 2))            
            new_k = torch.fft.ifft2(new_k, dim=(1, 2))
            new_k = math.sqrt(nsize[2]*nsize[3])*torch.fft.fftshift(new_k, dim=(1, 2))
            new_k = torch.stack((torch.real(new_k), torch.imag(new_k)), dim=1)            
            y1 = y + self.t*(new_k - or_im) # t learnable parameter  ## only keep the measured data
            x[:,0,:,:].data = y1[:,0,:,:]
            x[:,3,:,:].data = y1[:,1,:,:]         

# ################################### 
            # test_im = torch.complex(x[:, 0, :, :], x[:, 1, :, :])
            # # test_k = test_im
            # test_k = torch.fft.ifftshift(test_im, dim=(1, 2))
            # test_k = torch.fft.fft2(test_k, dim=(1, 2))
            # test_k = 1/math.sqrt(nsize[2]*nsize[3])*torch.fft.fftshift(test_k, dim=(1, 2))

            # test_k_ = test_k
            # test_k_ = test_k_.cpu()
            # plt.figure(10)
            # plt.imshow(np.log(abs(test_k_.detach_().numpy()[2, :, :])-1e-10), cmap='gray')
            # plt.savefig('iter_'+str(i)+'_kspace_after_soft_DC')

            # x_ = torch.complex(x[:, 0, :, :], x[:, 1, :, :])
            # x_ = x_.cpu()
            # plt.figure(61)
            # plt.imshow(abs(x_.detach_().numpy()[2, :, :]), cmap='gray')
            # plt.savefig('iter_'+str(i)+'_img_after_soft_DC_input')
            
###################################
#  
            res = self.head(x)
###################################            
            # test_im = res
            # test_im = torch.complex(res[:, :64, :, :], res[:, 64:, :, :])
            # # test_k = test_im
            # test_k = torch.fft.fftshift(test_im, dim=(2, 3))
            # test_k = torch.fft.fft2(test_k, dim=(2, 3))
            # test_k = 1/math.sqrt(nsize[2]*nsize[3])*torch.fft.fftshift(test_k, dim=(2, 3))
            # res_dis = test_im.permute(1,0,2,3)
            # slice1 = torch.reshape(res_dis[:,2,:,:],(64,1,256,256))
            # torchvision.utils.save_image(abs(slice1),'iter_'+str(i)+'_img_head_output.png',normalize=True)

            # res_dis = test_k.permute(1,0,2,3)
            # slice1 = torch.reshape(res_dis[:,2,:,:],(64,1,256,256))
            # torchvision.utils.save_image(torch.log(abs(slice1)),'iter_'+str(i)+'_k_head_output.png',normalize=True)
###################################  




            res = self.body(res)

# ###################################              
            # test_im = torch.complex(res[:, :64, :, :], res[:, 64:, :, :])
            # # test_k = test_im
            # test_k = torch.fft.ifftshift(test_im, dim=(2, 3))
            # test_k = torch.fft.fft2(test_k, dim=(2, 3))
            # test_k = 1/math.sqrt(nsize[2]*nsize[3])*torch.fft.fftshift(test_k, dim=(2, 3))
            # res_dis = test_im.permute(1,0,2,3)
            # slice1 = torch.reshape(res_dis[:,2,:,:],(64,1,256,256))
            # torchvision.utils.save_image(abs(slice1),'iter_'+str(i)+'_img_body_output.png',normalize=True,scale_each = True)

            # res_dis = test_k.permute(1,0,2,3)
            # slice1 = torch.reshape(res_dis[:,2,:,:],(64,1,256,256))
            # torchvision.utils.save_image(torch.log(abs(slice1)),'iter_'+str(i)+'_k_body_output.png',normalize=True,scale_each = True)


            # y = self.tail(res)
            # test_im = torch.complex(y[:, 0, :, :], y[:, 1, :, :])
            # # test_k =test_im
            # test_k = torch.fft.fftshift(test_im, dim=(1, 2))
            # test_k = torch.fft.fft2(test_k, dim=(1, 2))
            # test_k = 1/math.sqrt(nsize[2]*nsize[3])*torch.fft.fftshift(test_k, dim=(1, 2))

            # test_k_ = test_k
            # test_k_ = test_k_.cpu()
            # plt.figure(10)
            # plt.imshow(
            # np.log(abs(test_k_.detach_().numpy()[2, :, :])-1e-10), cmap='gray')
            # plt.savefig('iter_'+str(i)+'_kspace_tail_output_residual')            


            # y_ = torch.complex(y[:, 0, :, :], y[:, 1, :, :])
            # y_ = y_.cpu()
            # plt.figure(71)
            # plt.imshow(abs(y_.detach_().numpy()[2, :, :]), cmap='gray')
            # plt.savefig('iter_'+str(i)+'_img_tail_output_residual')
###################################     
            y = self.tail(res) + y   
#####################################            

        #     test_im = torch.complex(y[:, 0, :, :], y[:, 1, :, :])
        #     test_k = test_im
        #     test_k = torch.fft.ifftshift(test_im, dim=(1, 2))
        #     test_k = torch.fft.fft2(test_k, dim=(1, 2))
        #     test_k = 1/math.sqrt(nsize[2]*nsize[3])*torch.fft.fftshift(test_k, dim=(1, 2))
        #     test_k_ = test_k
        #     test_k_ = test_k_.cpu()
        #     plt.figure(10)
        #     plt.imshow(np.log(abs(test_k_.detach_().numpy()[2, :, :])-1e-10), cmap='gray')
        #     plt.savefig('iter_'+str(i)+'_kspace_tail_output')


        #     y_ = torch.complex(y[:, 0, :, :], y[:, 1, :, :])
        #     y_ = y_.cpu()
        #     plt.figure(71)
        #     plt.imshow(abs(y_.detach_().numpy()[2, :, :]), cmap='gray')
        #     plt.savefig('iter_'+str(i)+'_img_tail_output')

        # plt.figure(62)
        # plt.imshow(abs(y_.detach_().numpy()[2, :, :]), cmap='gray')
        # plt.savefig('img_before_hard_DC')
###################################
        new_k = torch.complex(y[:, 0, :, :], y[:, 1, :, :])
        new_k = torch.fft.ifftshift(new_k, dim=(1, 2))
        new_k = torch.fft.fft2(new_k, dim=(1, 2))
        new_k = 1/math.sqrt(nsize[2]*nsize[3])*torch.fft.fftshift(new_k, dim=(1, 2))
###################################
        # new_k_ = new_k
        # new_k_ = new_k_.cpu()
        # plt.figure(62)
        # plt.imshow(np.log(abs(new_k_.detach_().numpy()[2, :, :])), cmap='gray')
        # plt.savefig('k_before_hard_DC')
###################################
        new_k[:, :, pf_com:] = or_k[:, :, pf_com:]
###################################
        # new_k_ = new_k
        # new_k_ = new_k_.cpu()
        # plt.figure(62)
        # plt.imshow(np.log(abs(new_k_.detach_().numpy()[2, :, :])), cmap='gray')
        # plt.savefig('k_after_hard_DC')
###################################
        new_k = torch.fft.ifftshift(new_k, dim=(1, 2))
        new_k = torch.fft.ifft2(new_k, dim=(1, 2))
        new_k = math.sqrt(nsize[2]*nsize[3])*torch.fft.fftshift(new_k, dim=(1, 2))
###################################
        # new_k_ = new_k
        # new_k_ = new_k_.cpu()
        # plt.figure(6)
        # plt.imshow(abs(new_k_.detach_().numpy()[2, :, :]), cmap='gray')
        # plt.savefig('img_after_hard_DC')
###################################
        y = torch.stack((torch.real(new_k), torch.imag(new_k)), dim=1)
        return y

class Net_cpx_2D(nn.Module):
    def __init__(self):
        super(Net_cpx_2D, self).__init__()

        n_resblocks = 16
        n_feats = 32

        kernel_size = 3
        self.scale = 1
        n_colors = 1

        # act = ComplexReLU()
        # act = nn.ReLU(True)
        # self.sub_mean = MeanShift(rgb_range)
        # self.add_mean = MeanShift(rgb_range, sign=1)

        # define head module
        # m_head = ComplexConv2d(n_colors, n_feats, kernel_size)

        # define body module
        m_body = [
            ResBlock(n_feats, kernel_size,  res_scale=1
                     ) for _ in range(n_resblocks)
        ]
        m_body.append(ComplexConv2d(n_feats, n_feats, kernel_size))

        # define tail module

        # m_tail = [
        #     ComplexConv2d(n_feats, n_colors, kernel_size)
        # ]

        self.head = ComplexConv2d(n_colors, n_feats, kernel_size)
        self.body = nn.Sequential(*m_body)
        self.tail = ComplexConv2d(n_feats, n_colors, kernel_size)

        # self.cc = ComplexConv2d(1,64)

        self.register_parameter("t", nn.Parameter(-2*torch.ones(1)))

    def forward(self, x):
        or_im = x
        nsize =x.size()
        pf_1= math.floor(nsize[3]*0.40)
        pf_com_1 =nsize[3] -pf_1
        pf_0= math.floor(nsize[2]*0.40)
        pf_com_0 =nsize[2] -pf_0        

        # print(nsize[3])
        # pf_ratio =torch.int(torch.floor(nsize[3]*0.45))
        or_k = torch.complex(or_im[:, 0, :, :], or_im[:, 1, :, :])
        or_k = torch.fft.ifftshift(or_k, dim=(1, 2))
        or_k = torch.fft.fft2(or_k, dim=(1, 2))
        or_k = 1/math.sqrt(nsize[2]*nsize[3])*torch.fft.fftshift(or_k, dim=(1, 2))

######################################################
        # or_k_ = or_k
        # or_k_ = or_k_.cpu()
        # plt.figure(6)
        # plt.imshow(np.log(abs(or_k_.detach_().numpy()[2, :, :])-1e-10), cmap='gray')
        # plt.savefig('input_k_or')

        # or_im_ = torch.complex(or_im[:,0,:,:],or_im[:,1,:,:])
        # or_im_ = or_im_.cpu()
        # plt.figure(7)
        # plt.imshow((abs(or_im_.detach_().numpy()[2, :, :])), cmap='gray')
        # plt.savefig('input_img_or')
#################################################

        y = x
        for i in range(2):
            x = y
            new_k = torch.complex(x[:, 0, :, :], x[:, 1, :, :])

###################################            
            # new_k_ = new_k
            # new_k_ = new_k_.cpu()
            # plt.figure(70)
            # plt.imshow((abs(new_k_.detach_().numpy()[2, :, :])), cmap='gray')
            # plt.savefig('iter_'+str(i)+'_img_before_DC')
################################### 
            new_k = torch.fft.ifftshift(new_k, dim=(1,2))
            new_k = torch.fft.fft2(new_k, dim=(1, 2))
            new_k = 1/math.sqrt(nsize[2]*nsize[3])*torch.fft.fftshift(new_k, dim=(1,2))

################################### 
            # new_k_ = new_k
            # new_k_ = new_k_.cpu()
            # plt.figure(80)
            # plt.imshow(np.log(abs(new_k_.detach_().numpy()[2, :, :])-1e-10), cmap='gray')
            # plt.savefig('iter_'+str(i)+'_kspace_before_DC')

################################### 
            new_k[:, :pf_0, :] = or_k[:, :pf_0, :]  # only keep the measured data
            new_k[:, :, :pf_1] = or_k[:, :, :pf_1]  # only keep the measured data

            new_k = torch.fft.ifftshift(new_k, dim=(1, 2))            
            new_k = torch.fft.ifft2(new_k, dim=(1, 2))
            new_k = math.sqrt(nsize[2]*nsize[3])*torch.fft.fftshift(new_k, dim=(1, 2))
            new_k = torch.stack((torch.real(new_k), torch.imag(new_k)), dim=1)            
            x = x + self.t*(new_k - or_im) # t learnable parameter  ## only keep the measured data

# ################################### 
            # test_im = torch.complex(x[:, 0, :, :], x[:, 1, :, :])
            # # test_k = test_im
            # test_k = torch.fft.ifftshift(test_im, dim=(1, 2))
            # test_k = torch.fft.fft2(test_k, dim=(1, 2))
            # test_k = 1/math.sqrt(nsize[2]*nsize[3])*torch.fft.fftshift(test_k, dim=(1, 2))

            # test_k_ = test_k
            # test_k_ = test_k_.cpu()
            # plt.figure(10)
            # plt.imshow(np.log(abs(test_k_.detach_().numpy()[2, :, :])-1e-10), cmap='gray')
            # plt.savefig('iter_'+str(i)+'_kspace_after_soft_DC')

            # x_ = torch.complex(x[:, 0, :, :], x[:, 1, :, :])
            # x_ = x_.cpu()
            # plt.figure(61)
            # plt.imshow(abs(x_.detach_().numpy()[2, :, :]), cmap='gray')
            # plt.savefig('iter_'+str(i)+'_img_after_soft_DC_input')
            
###################################
#  
            res = self.head(x)
###################################            
            # test_im = res
            # test_im = torch.complex(res[:, :64, :, :], res[:, 64:, :, :])
            # # test_k = test_im
            # test_k = torch.fft.fftshift(test_im, dim=(2, 3))
            # test_k = torch.fft.fft2(test_k, dim=(2, 3))
            # test_k = 1/math.sqrt(nsize[2]*nsize[3])*torch.fft.fftshift(test_k, dim=(2, 3))
            # res_dis = test_im.permute(1,0,2,3)
            # slice1 = torch.reshape(res_dis[:,2,:,:],(64,1,256,256))
            # torchvision.utils.save_image(abs(slice1),'iter_'+str(i)+'_img_head_output.png',normalize=True)

            # res_dis = test_k.permute(1,0,2,3)
            # slice1 = torch.reshape(res_dis[:,2,:,:],(64,1,256,256))
            # torchvision.utils.save_image(torch.log(abs(slice1)),'iter_'+str(i)+'_k_head_output.png',normalize=True)
###################################  




            res = self.body(res)

# ###################################              
            # test_im = torch.complex(res[:, :64, :, :], res[:, 64:, :, :])
            # # test_k = test_im
            # test_k = torch.fft.ifftshift(test_im, dim=(2, 3))
            # test_k = torch.fft.fft2(test_k, dim=(2, 3))
            # test_k = 1/math.sqrt(nsize[2]*nsize[3])*torch.fft.fftshift(test_k, dim=(2, 3))
            # res_dis = test_im.permute(1,0,2,3)
            # slice1 = torch.reshape(res_dis[:,2,:,:],(64,1,256,256))
            # torchvision.utils.save_image(abs(slice1),'iter_'+str(i)+'_img_body_output.png',normalize=True,scale_each = True)

            # res_dis = test_k.permute(1,0,2,3)
            # slice1 = torch.reshape(res_dis[:,2,:,:],(64,1,256,256))
            # torchvision.utils.save_image(torch.log(abs(slice1)),'iter_'+str(i)+'_k_body_output.png',normalize=True,scale_each = True)


            # y = self.tail(res)
            # test_im = torch.complex(y[:, 0, :, :], y[:, 1, :, :])
            # # test_k =test_im
            # test_k = torch.fft.fftshift(test_im, dim=(1, 2))
            # test_k = torch.fft.fft2(test_k, dim=(1, 2))
            # test_k = 1/math.sqrt(nsize[2]*nsize[3])*torch.fft.fftshift(test_k, dim=(1, 2))

            # test_k_ = test_k
            # test_k_ = test_k_.cpu()
            # plt.figure(10)
            # plt.imshow(
            # np.log(abs(test_k_.detach_().numpy()[2, :, :])-1e-10), cmap='gray')
            # plt.savefig('iter_'+str(i)+'_kspace_tail_output_residual')            


            # y_ = torch.complex(y[:, 0, :, :], y[:, 1, :, :])
            # y_ = y_.cpu()
            # plt.figure(71)
            # plt.imshow(abs(y_.detach_().numpy()[2, :, :]), cmap='gray')
            # plt.savefig('iter_'+str(i)+'_img_tail_output_residual')
###################################     
            y = self.tail(res) + x   
#####################################            

        #     test_im = torch.complex(y[:, 0, :, :], y[:, 1, :, :])
        #     test_k = test_im
        #     test_k = torch.fft.ifftshift(test_im, dim=(1, 2))
        #     test_k = torch.fft.fft2(test_k, dim=(1, 2))
        #     test_k = 1/math.sqrt(nsize[2]*nsize[3])*torch.fft.fftshift(test_k, dim=(1, 2))
        #     test_k_ = test_k
        #     test_k_ = test_k_.cpu()
        #     plt.figure(10)
        #     plt.imshow(np.log(abs(test_k_.detach_().numpy()[2, :, :])-1e-10), cmap='gray')
        #     plt.savefig('iter_'+str(i)+'_kspace_tail_output')


        #     y_ = torch.complex(y[:, 0, :, :], y[:, 1, :, :])
        #     y_ = y_.cpu()
        #     plt.figure(71)
        #     plt.imshow(abs(y_.detach_().numpy()[2, :, :]), cmap='gray')
        #     plt.savefig('iter_'+str(i)+'_img_tail_output')

        # plt.figure(62)
        # plt.imshow(abs(y_.detach_().numpy()[2, :, :]), cmap='gray')
        # plt.savefig('img_before_hard_DC')
###################################
        new_k = torch.complex(y[:, 0, :, :], y[:, 1, :, :])
        new_k = torch.fft.ifftshift(new_k, dim=(1, 2))
        new_k = torch.fft.fft2(new_k, dim=(1, 2))
        new_k = 1/math.sqrt(nsize[2]*nsize[3])*torch.fft.fftshift(new_k, dim=(1, 2))
###################################
        # new_k_ = new_k
        # new_k_ = new_k_.cpu()
        # plt.figure(62)
        # plt.imshow(np.log(abs(new_k_.detach_().numpy()[2, :, :])), cmap='gray')
        # plt.savefig('k_before_hard_DC')
###################################
        or_k_ = or_k
        or_k_[:, :pf_0, :] = new_k[:, :pf_0, :]  # only keep the measured data
        or_k_[:, :, :pf_1] = new_k[:, :, :pf_1]  # only keep the measured data
        new_k = or_k_
###################################
        # new_k_ = new_k
        # new_k_ = new_k_.cpu()
        # plt.figure(62)
        # plt.imshow(np.log(abs(new_k_.detach_().numpy()[2, :, :])), cmap='gray')
        # plt.savefig('k_after_hard_DC')
###################################
        new_k = torch.fft.ifftshift(new_k, dim=(1, 2))
        new_k = torch.fft.ifft2(new_k, dim=(1, 2))
        new_k = math.sqrt(nsize[2]*nsize[3])*torch.fft.fftshift(new_k, dim=(1, 2))
###################################
        # new_k_ = new_k
        # new_k_ = new_k_.cpu()
        # plt.figure(6)
        # plt.imshow(abs(new_k_.detach_().numpy()[2, :, :]), cmap='gray')
        # plt.savefig('img_after_hard_DC')
###################################
        y = torch.stack((torch.real(new_k), torch.imag(new_k)), dim=1)
        return y
