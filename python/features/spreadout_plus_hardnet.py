"""
OpenCV KAZE Implementation
Author: Alex Butenko
"""

from features.DetectorDescriptorTemplate import DetectorAndDescriptor
import sys
import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
import features.feature_utils as utils
import time

dirname = os.path.dirname(__file__)
MAX_BATCH = 1024
class spreadout_plus_hardnet(DetectorAndDescriptor):
    def __init__(self):
        super(
            spreadout_plus_hardnet,
            self).__init__(
                name='spreadout_plus_hardnet',
                is_detector=False,
                is_descriptor=True,
                is_both=False,
                patch_input=True)

        cudnn.benchmark = True
        self.model = HardNetModel()
        self.device = ('cuda' if torch.cuda.is_available() else 'cpu')
        ckpt = torch.load(os.path.join(dirname,'spreadout_plus_hardnet_misc/checkpoint_9_with_gor.pth'),
                          map_location=self.device)

        self.model.load_state_dict(ckpt['state_dict'])
        self.model.eval()


    def extract_descriptors_from_patch_batch(self, batch):
        gray_batch = list()
        for  p in batch:
            gray_batch.append(utils.all_to_gray(p))

        batch = np.array(gray_batch)
        batch = np.expand_dims(batch, axis=1)
        #make multiple batches of Max batch size
        splits = np.arange(0,batch.shape[0],MAX_BATCH)[1:]
        batch_list = np.split(batch, splits)
        descriptor_list = list()

        for bat in batch_list:
            if bat.size==0:
                continue
            bat = torch.from_numpy(bat)
            descriptors = self.model(bat.float())
            descriptor_list.extend(descriptors.detach().numpy().tolist())

        descriptors = np.array(descriptor_list)
        return descriptors

    def extract_descriptor(self, image, feature):
        """
        feature = [[x, y, scale, angle]...]
        """
        gray_image = utils.all_to_gray(image)
        patches = []
        print('extracting {} patches'.format(len(feature)))
        for f in feature:
            patch = utils.extract_patch(gray_image, f, patch_sz=32)
            patches.append(patch)
        patches = np.array(patches)

        desc = self.extract_descriptors_from_patch_batch(patches)

        return desc


class HardNetModel(nn.Module):
    """HardNet model definition"""
    def __init__(self):
        super(HardNetModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias = False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias = False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias = False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias = False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2,padding=1, bias = False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias = False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(128, 128, kernel_size=8, bias = False),
            nn.BatchNorm2d(128, affine=False),
        )

    def input_norm(self,x):
        flat = x.view(x.size(0), -1)
        mp = torch.sum(flat, dim=1) / (32. * 32.)
        sp = torch.std(flat, dim=1) + 1e-7
        return (x - mp.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)) / sp.unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x)

    def forward(self, input):
        norm_x = self.input_norm(input)
        x_features = self.features(norm_x)
        x = x_features.view(x_features.size(0), -1)
        return L2Norm()(x)

class L2Norm(nn.Module):
    def __init__(self):
        super(L2Norm,self).__init__()
        self.eps = 1e-10
    def forward(self, x):
        norm = torch.sqrt(torch.sum(x * x, dim = 1) + self.eps)
        x= x / norm.unsqueeze(-1).expand_as(x)
        return x
