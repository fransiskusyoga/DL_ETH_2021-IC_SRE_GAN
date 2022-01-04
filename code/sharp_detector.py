import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair, _quadruple
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

class SharpDetector:
    def __init__(self):
        # --- initialize filters ---
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            raise ValueError("cuda is unavailable now.")
        self.laplacian_filter = torch.FloatTensor(
            [[0, 1, 0], [1, -4, 1], [0, 1, 0]]).view(1, 1, 3, 3).to(self.device)
        
    def get_mask(self, image):
        s = 1
        pad = 1
        gray = self.getGrayImage(image)
        img_lap = torch.nn.functional.conv2d(input=gray,
                                        weight=Variable(self.laplacian_filter),
                                        stride=s,
                                        padding=pad)
        out = self.blurring(img_lap)
        return out 

    def getGrayImage(self, rgbImg):
        gray = (rgbImg[:,0,:,:] + rgbImg[:,1,:,:] + rgbImg[:,2,:,:]) / 3.0
        gray = torch.unsqueeze(gray, 1)
        return gray

    def blurring(self, laplased_image, sigma=5, min_abs=0.5/255):
        # --- sigma is the size of kernel of Blurr filter ---
        abs_image = torch.abs(laplased_image).to(torch.float32)  # convert to absolute values
        abs_image[abs_image < min_abs] = min_abs 
        # print(sigma)
        blurred_img = self.BlurLayer(abs_image, k_size=sigma)
        return blurred_img

    def BlurLayer(self, img, k_size=5, s=1, pad=2):
        _blur_filter = torch.ones([k_size, k_size]).to(self.device)
        blur_filter = _blur_filter.view(1,1,k_size, k_size) / (k_size**2)
        # gray = getGrayImage(img)
        img_blur = torch.nn.functional.conv2d(input=img,
                                            weight=Variable(blur_filter),
                                            stride=s,
                                            padding=pad)

        return img_blur

if __name__=="__main__":
    # --- 画像のロード ---
    # img_id = 5
    img_path = "Black_Footed_Albatross_0001_796111.jpg"
    img_path = "Yellow_Bellied_Flycatcher_0045_42575.jpg"
    im = np.array(Image.open(img_path))
    
    img = torch.tensor(im) / 255.0
    img = torch.tensor(img.transpose(1, 2).transpose(0, 1))
    # img = img.transpose(0,2)  # 転置
    shp = img.shape
    img = img.view(1, *shp)  # バッチ方向を作成、ここ普通にmax255
    # ---- バッチ方向にrepeat ----
    #img = torch.repeat_interleave(img, dim=0, repeats=3)
    img = torch.mean(img,dim=1).unsqueeze(1)
    print("img shape: ", img.shape)

    laplacian_filter = torch.FloatTensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]]).view(1, 1, 3, 3)
    blur_filter = torch.FloatTensor([[1,2, 1], [2, 4, 2], [1, 2, 1]]).view(1, 1, 3, 3)
    blur_filter = blur_filter/blur_filter.sum()

    out = torch.nn.functional.pad(img,(1,1,1,1),mode='replicate')
    out = torch.nn.functional.conv2d(input=out, weight=Variable(laplacian_filter), stride=1, padding=0) # the gradient image
    out = torch.nn.functional.conv2d(input=out, weight=Variable(blur_filter), stride=1, padding=0)
    abs_out = torch.abs(out)

    plt.hist(abs_out.flatten().detach().numpy(), bins =100)
    plt.yscale('log')
    minor_ticks = np.arange(0, 1, 0.01)
    # plt.xticks(minor_ticks,minor=True)
    plt.axes().xaxis.set_minor_locator(MultipleLocator(0.01))
    plt.grid(which='both', axis='both')
    plt.show()

    # # method 1s
    # abs_out = torch.abs(out)
    # sort_out,_ = torch.sort(abs_out.view(-1), descending=True)
    # cum_val = torch.cumsum(sort_out,dim=0)
    # cum_val = cum_val/cum_val[-1]
    # val = sort_out[torch.sum(cum_val<0.5)]
    # print("AAAAAAAAAAAAAAAAA", val)
    # plt.plot(sort_out,cum_val)
    # plt.grid()
    # plt.show()

    # method 2
    val = torch.mean(abs_out)
    for i in range(5):
        mask = abs_out>val
        m1 = torch.mean(abs_out[mask])
        mask = torch.logical_not(mask)
        m2 = torch.mean(abs_out[mask])
        val = (m1+m2)/2.0
        print(val)

    # mask1 = F.max_pool2d(out,kernel_size=3, stride=1,padding=1)==out
    # mask2 = F.max_pool2d(-out,kernel_size=3, stride=1,padding=1)==(-out)
    # np_out = torch.logical_or(mask1,mask2)*(abs_out>=torch.mean(abs_out)*5)
    np_out = (abs_out>=val)
    np_out = np_out * torch.max(abs_out)
    np_out = np.concatenate((np_out, abs_out),axis=3)
    plt.imshow(np_out[0, 0,:,:], cmap='gray')
    # plt.savefig('./blurred_test.png')
    plt.show()
    plt.clf()