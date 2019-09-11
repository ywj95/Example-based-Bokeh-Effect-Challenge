from __future__ import print_function
import argparse
import os
import time
import torch
import numpy as np
import scipy.misc
import imageio
from PIL import Image

import cv2
from skimage.measure import compare_psnr as psnr
from skimage.measure import compare_ssim as ski_ssim


def norm_ip(img, min, max):
    img.clamp_(min=min, max=max)
    img.add_(-min).div_(max - min)

    return img


def norm_range(t, range):
    if range is not None:
        norm_ip(t, range[0], range[1])
    else:
        norm_ip(t, t.min(), t.max())
    return norm_ip(t, t.min(), t.max())

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")


def test(Ix, model):
    med_time = []

    with torch.no_grad():
        Ix = Ix.to(device)

        start_time = time.perf_counter()  # -------------------------begin to deal with an image's time

        Ix_cc = model(Ix)
        '''
        tensor = norm_range(Ix_cc[0].cpu(), None)
        ndarr = tensor.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
        '''
        # modify
        # tensor = norm_range(torch.squeeze(Ix_cc), None)
        # ndarr = tensor.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()

        Ix_cc = Ix_cc[0].clamp(0, 1)
        Ix_cc = np.uint8(255 * Ix_cc.permute(1, 2, 0).cpu().numpy())
        torch.cuda.synchronize()  # wait for CPU & GPU time syn

        evalation_time = time.perf_counter() - start_time  # ---------finish an image
        med_time.append(evalation_time)
        print('time',evalation_time)  # add
    return Ix_cc


# models/modelOut/5/GFNMS2_2_epoch_14.pklmodels/modelOut/10/GFNMS2_2_epoch_11.pkl
def print_network(net):
    if isinstance(net, list):
        net = net[0]
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

# for i in range(28, 30):
from network.Res29_1 import Ensemble
model = Ensemble().to(device)
model.load_state_dict(torch.load("/home/ywj/game/models/modelOut/MS58/Dem_60.pkl"))

#im_path = '/home/ywj/game/dataset/ValidationBokehFree/'
im_path = '/home/zcc/Samples/broken/game-new/TestBokehFree/'


ave_psnr = 0.0
ave_ssim = 0.0
for path, subdirs, files in os.walk(im_path):
    for i in range(len(files)):
        nameA = files[i]
        hazyName = im_path + nameA

        print(hazyName)
        Ix = np.array(Image.open(hazyName).convert('RGB'))
        print('Ix',Ix.shape)  #add
        H = Ix.shape[1]
        W = Ix.shape[0]
        img_H = H // 8 * 8
        img_W = W // 8 * 8
        Ixin = cv2.resize(Ix, (img_H, img_W))

        model = model.to(device)
        testI = Ixin
        testI = np.array([testI])
        print('testI1',testI.shape)  #add
        testI = testI.transpose([0, 3, 1, 2])
        print('testI2',testI.shape)  #add
        testI = torch.from_numpy(testI).float() / 255.


        out = test(testI, model).astype(np.uint8)


        out = cv2.resize(out, (H, W)).astype(np.uint8)


        imageio.imwrite('///home/ywj/game/Training/result/' + nameA, out.astype(np.uint8))
        #scipy.misc.imsave('/media/leo/data/hm/GFN-dehazing-master/out_our/' + nameA,
        #                  np.hstack([Ix, out.astype(np.uint8), np.abs(Ix - out.astype(np.uint8))]))



