#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 28 16:45:56 2021

@author: zhye
"""
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import time
from skimage.exposure import match_histograms
from skimage.morphology import area_opening, area_closing, dilation, disk
from skimage.filters import gaussian

def read_image(img_name):

    img = np.array(Image.open(os.path.join(data_path, img_name)))

    return img

def norm(img, rng=(0, 255)):
    r_min = img.min()
    r_max = img.max()
    t_min = rng[0]
    t_max = rng[1]

    return (img - r_min) / (r_max - r_min) * (t_max-t_min) + t_min


def get_channel_diff(real, fake, channel=0, threshold='manual', set_thresh=None):

    real = gaussian(real, sigma=1, multichannel = True) * 255.0
    fake = gaussian(fake, sigma=1, multichannel = True) * 255.0

    matched = match_histograms(real, fake).astype(int)

    real = matched[:, :, channel]
    fake = fake[:, :, channel]

    diff = real - fake

    abs_diff = np.absolute(diff)

    if threshold == 'median':
        thresh = np.median(abs_diff) * 4.5
    else:
        if set_thresh:
            thresh = set_thresh
        else:
            thresh = 30 # 30

    out_diff = (abs_diff >= thresh).astype(int)

    return out_diff

def get_diff(real, fake, threshold='manual', set_thresh=None):

    ch0 = get_channel_diff(real, fake, channel=0, threshold=threshold, set_thresh=set_thresh)
    ch1 = get_channel_diff(real, fake, channel=1, threshold=threshold, set_thresh=set_thresh)
    ch2 = get_channel_diff(real, fake, channel=2, threshold=threshold, set_thresh=set_thresh)

#     print()
#     print('-' * 25)
#     print()

    ch = ch0 + ch1 + ch2
    ch = (ch > 0).astype(int)

    ch = area_opening(ch, area_threshold=200)
    ch = area_closing(ch, area_threshold=500)

    ch = dilation(ch, disk(5))

    return ch

def get_seg(real, fake):

    score, diff = ssim(real, fake, full=True, multichannel=True)

    diff_norm = norm(diff)

    diff_ch_sum = diff_norm[:,:,0].astype(int) + diff_norm[:,:,1].astype(int) + diff_norm[:,:,2].astype(int)

    diff_ch_norm = norm(diff_ch_sum)

    #plt.imshow(diff_ch_norm, cmap='gray')

    seg = chan_vese(diff_ch_norm)

    return seg


def invert(mat):
    return 1 - mat


def get_img_mask(real, fake, show=False, threshold='manual', set_thresh=None):
    diff = get_diff(real, fake, threshold=threshold, set_thresh=set_thresh)
    return np.invert((diff).astype(bool))

with open('/cluster/home/yezh/DL/img_name.txt') as f:
    img_name = f.readlines()

score = []

for i in range(len(img_name)):
    data_path = "/cluster/home/yezh/DL/Lightweight-Manipulation/output/birds_lightweight_2021_12_27_15_35_26/Model/netG_epoch_150/valid"
    real = read_image('real_r' + str(img_name[i][8:-1]))
    fake = read_image('fake_s' + str(img_name[i][8:-1]))
    mask = get_img_mask(fake, real, set_thresh=30)
    real_mask = real*mask[...,None]
    fake_mask = fake*mask[...,None]
    score.append(np.mean(np.square(real_mask - fake_mask)))
    #print("\r Process{0}%".format(round((i+1)*100/1000)), end="")
print(np.mean(score))
np.save('score.npy', np.array(score))

