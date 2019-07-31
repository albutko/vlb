"""
OpenCV VGG Convex Optimization Implementation
Author: Alex Butenko
"""
import cv2
import numpy as np
from features.DetectorDescriptorTemplate import DetectorAndDescriptor
import sys


class cv_convopt(DetectorAndDescriptor):
    def __init__(self):
        super(
            cv_convopt,
            self).__init__(
                name='cv_convopt',
                is_detector=False,
                is_descriptor=True,
                is_both=False,
                patch_input=False)

    def extract_descriptor(self, image, features):
        keypoints = list()
        for feature in features:
            kpt = cv2.KeyPoint()
            kpt.pt = (feature[0], feature[1])
            kpt.size = feature[2]
            kpt.angle = feature[3]
            keypoints.append(kpt)

        vgg = cv2.xfeatures2d.VGG_create()

        _, descriptors = vgg.compute(image, keypoints)

        return descriptors

    def extract_descriptor_from_patch(self, patches):
        pass
