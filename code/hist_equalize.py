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
import time

def hist_equalizer(img, min = 0.0, max = 1.0, int_out=False):
    """
    the input should have this dimension [*,H,W]
    * = can be any tensor size such as [Batch,Channel] or just [Channel]
    H = height
    W = weight
    H and W of ref and img may not match but the min,max, and bins across 
    all batch and channel should be the same
    """
    if int_out:
        max = max + 1
    # reshaping image for easier handling
    shp = img.shape
    img = img.view(-1,shp[-2]*shp[-1])
    # sorting as histogram subtitution
    sort_val,sort_idx = torch.sort(img, descending=False)
    # get the new value of the pixel
    sort_new_val = torch.linspace(min,max,shp[-1]*shp[-2]).to(img.device)
    n = round(img.nelement()/(shp[-2]*shp[-1]))
    sort_new_val = sort_new_val.repeat((n,1),1)
    # insert the new value to the 
    img_new = torch.zeros_like(img)
    img_new.scatter_(1, sort_idx,sort_new_val)
    img_new = img_new.view(*shp)
    if int_out:
        img_new.floor_()
    return img_new

def matching_histogram(img, ref, min = 0.0, max = 1.0, int_out=False):
    """
    the input should have this dimension [*,H,W]
    * = can be any tensor size such as [Batch,Channel] or just [Channel]
    H = height
    W = weight
    H and W of ref and img may not match but the min,max, and bins across 
    all batch and channel should be the same
    """
    if int_out:
        max = max + 1
    # reshape the image for easier handling
    img_shp = img.shape
    ref_shp = ref.shape
    img = img.view(-1,img_shp[-2]*img_shp[-1])
    ref = ref.view(-1,ref_shp[-2]*ref_shp[-1])
    # doing sorting instead of histogram
    sort_val,sort_idx = torch.sort(img, descending=False)
    sort_new_val,_ = torch.sort(ref, descending=False)
    # normalize the image size if it is different
    if (img_shp!=ref_shp):
        max = ref_shp[-2]*ref_shp[-1]
        step = max / (img_shp[-2]*img_shp[-1])
        idx = torch.arange(0,max,step).to(torch.int).to(img.device)
        sort_new_val = torch.index_select(sort_new_val, 1 , idx)
    # inserting the value
    img_new = torch.zeros_like(img)
    img_new.scatter_(1, sort_idx,sort_new_val)
    img_new = img_new.reshape(*img_shp)
    if int_out:
        img_new.floor_()
    return img_new

# def matching_histogram2(img, ref, min=0, max=255, bins=256):
#     """
#     the input should have this dimension [*,H,W]
#     * = can be any tensor size such as [Batch,Channel] or just [Channel]
#     H = height
#     W = weight
#     H and W of ref and img may not match but the min,max, and bins across 
#     all batch and channel should be the same
#     """
#     step = (max-min)/(bins-1)
#     img_shp = img.shape
#     img = img.view(-1,img_shp[2]*img_shp[3])
#     ref_shp = ref.shape
#     ref = ref.view(-1,ref_shp[2]*ref_shp[3])
#     # get histogram
#     img_hist = torch.stack([torch.histc(ti, bins=bins, min=min, max=max) for ti in img])
#     ref_hist = torch.stack([torch.histc(ti, bins=bins, min=min, max=max) for ti in ref])
#     #cumulative sum of histogram
#     img_hist = torch.cumsum(img_hist,dim=1) / (img_shp[2]*img_shp[3])
#     ref_hist = torch.cumsum(ref_hist,dim=1) / (ref_shp[2]*ref_shp[3])
#     # get new value
#     new_val = torch.sum( (img_hist[:,:,None] >= ref_hist[:,None,:]), dim=2)
#     new_val = new_val * step + min

#     img_new = torch.zeros_like(img)
#     for i in range(bins):
#         lower = step*i + min
#         upper = lower + step
#         mask = (img>=lower) & (img<upper)
#         img_new = img_new + mask*(new_val[:,i].unsqueeze(1))
    
#     # inc_val = torch.sum( (img_hist[:,:,None] >= ref_hist[:,None,:]), dim=2)
#     # inc_val = (inc_val[:,1:] - inc_val[:,0:-1]) * step 
#     # img_new = torch.linspace(min+step,max,bins-1).to(img.device)
#     # img_new = torch.sum( (img[:,:,None] >= img_new[None,None,:])*inc_val[:,None,:] , dim = 2) + min

#     return img_new.view(*img_shp)

if __name__=="__main__":
    # img_id = 5
    img_path = "Yellow_Bellied_Flycatcher_0045_42575.jpg"
    img = np.array(Image.open(img_path))
    img = torch.tensor(img)/1.0
    img = torch.tensor(img.transpose(1, 2).transpose(0, 1))
    img = img.view(1, *img.shape)  
    img = torch.repeat_interleave(img, dim=0, repeats=3).cuda()
    #img = torch.mean(img,dim=1).unsqueeze(1)

    
    img_path = "Black_Footed_Albatross_0001_796111.jpg"
    img2 = np.array(Image.open(img_path))
    img2 = torch.tensor(img2)/1.0
    img2 = torch.tensor(img2.transpose(1, 2).transpose(0, 1))
    img2 = img2.view(1, *img2.shape)  
    img2 = torch.repeat_interleave(img2, dim=0, repeats=3).cuda()
    #img2 = torch.mean(img2,dim=1).unsqueeze(1)

    start = time.time()
    for i in range(10):
        img = hist_equalizer(img, min = 0.0, max = 255.0, int_out=True)
        #img = matching_histogram(img2, img, min = 0.0, max = 255.0, int_out=True)
    print(time.time()-start)

    print("img shape: ", img.shape)
    plt.imshow(img[0, :,:,:].cpu().transpose(0,1).transpose(1,2)/255)
    # plt.savefig('./blurred_test.png')
    plt.show()
    plt.clf()