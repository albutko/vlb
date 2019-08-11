"""
IMIPS Implentation
Author: Alex Butenko
"""
import cv2
import numpy as np
from features.DetectorDescriptorTemplate import DetectorAndDescriptor
import features.feature_utils as fu
import sys
import os

dirname = os.path.dirname(__file__)

sys.path.append(dirname+'/imips_misc/imips_open/python')
sys.path.append(dirname+'/imips_misc/imips_open/python/imips')
sys.path.append(dirname+'/imips_misc/imips_open_deps/rpg_common_py/python')
sys.path.append(dirname+'/imips_misc/imips_open_deps/rpg_datasets_py/python')
import imips.hyperparams as hyperparams

MAX_KPTS = 1000
class IMIPS(DetectorAndDescriptor):
    def __init__(self):
        super(
            IMIPS,
            self).__init__(
            name='IMIPS',
            is_detector=True,
            is_descriptor=False,
            is_both=False,
            patch_input=False)

        graph, sess = hyperparams.bootstrapFromCheckpoint()
        self.forward_passer = hyperparams.getForwardPasser(graph, sess)

    def detect_feature(self, image):
        image = fu.all_to_gray(image)
        out = self.forward_passer(image)
        kpts = out.ips_rc.T
        print(kpts.shape)
        order = np.argsort(out.ip_scores)[::-1]

        num_kpts = MAX_KPTS
        if len(order) < MAX_KPTS:
            num_kpts = len(order)

        order = order[:num_kpts]
        kpts = kpts[order,:]
        kpts = kpts[:, [1,0]]

        return kpts
