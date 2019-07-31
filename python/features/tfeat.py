"""
TFeat Implementation
Author: Alex Butenko
"""
import sys
import os

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import cv2

from DetectorDescriptorTemplate import DetectorAndDescriptor
import feature_utils as utils


dirname = os.path.dirname(__file__)
class tfeat(DetectorAndDescriptor):
    def __init__(self, pretrained_model='tfeat_misc/tfeat-liberty.params'):
        super(
            tfeat,
            self).__init__(
            name='tfeat',
            is_detector=False,
            is_descriptor=True,
            is_both=False,
            patch_input=True,
            can_batch=True)

        self.model = TNet()
        pretrained_model = os.path.join(dirname, pretrained_model)
        self.model.load_state_dict(torch.load(pretrained_model, map_location='cpu'))
        self.model.eval()

    def extract_descriptors_from_patch_batch(self, batch):
        nb_patches = batch.shape[0]

        batch_resized = list()
        for i, patch in enumerate(batch):
            batch_resized.append(cv2.resize(batch[i], (32, 32), interpolation=cv2.INTER_AREA))

        batch_resized = torch.tensor(batch_resized)

        batch_resized = batch_resized.view(-1,1,32,32)
        desc = self.model(batch_resized.float())
        return desc.detach().numpy()

    def extract_descriptors_from_patch(self, patch):

        patch = cv2.resize(patch, (32, 32), interpolation=cv2.INTER_AREA)
        patch = torch.tensor(patch)

        patch = patch.view(1,1,32,32)
        desc = self.model(patch.float())
        return desc.detach().numpy()

    def extract_descriptor(self, image, feature):
        gray_image = utils.all_to_gray(image)
        patches = []
        for f in feature:
            patch = utils.extract_patch_cv(image, f, patch_sz=32)
            patches.append(patch)

        patches = np.array(patches)
        desc = self.extract_descriptors_from_patch_batch(patches)

        return desc


class TNet(nn.Module):
    """TFeat model definition
    """
    def __init__(self, pretrained_model=None):
        super(TNet, self).__init__()
        self.features = nn.Sequential(
            nn.InstanceNorm2d(1, affine=False),
            nn.Conv2d(1, 32, kernel_size=7),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=6),
            nn.Tanh()
        )
        self.descr = nn.Sequential(
            nn.Linear(64 * 8 * 8, 128),
            nn.Tanh()
        )


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.descr(x)
        return x
