"""
OpenCV SIFT Implementation
Author: Alex Butenko
"""
from .DetectorDescriptorTemplate import DetectorAndDescriptor

import cv2
import numpy as np

class cv_sift(DetectorAndDescriptor):
    def __init__(self):
        super(
            cv_sift,
            self).__init__(
                name='cv_sift',
                is_detector=True,
                is_descriptor=True,
                is_both=True,
                patch_input=True)


    def detect_feature(self, image):
        surf = cv2.xfeatures2d.SIFT_create()
        features =  surf.detect(image, None)
        pts = np.array([features[idx].pt for idx in range(len(features))])
        return pts

    def extract_descriptor(self, image, feature):
        surf = cv2.xfeatures2d.SIFT_create()
        descriptors =  surf.compute(image, feature)
        return descriptors

    def extract_all(self, image):
        surf = cv2.xfeatures2d.SIFT_create()
        features, descriptors =  surf.detectAndCompute(image, None)
        pts = np.array([features[idx].pt for idx in range(len(features))])
        return (pts, descriptors)

    def extract_descriptor_from_patch(self, patch):
        sift = cv2.xfeatures2d.SIFT_create()
        ckp = get_center_kp()
        _, descriptor = sift.compute(patch.astype(np.uint8), [ckp])
        return descriptor


def get_center_kp(PS=65.):
    c = PS/2.0
    center_kp = cv2.KeyPoint()
    center_kp.pt = (c,c)
    center_kp.size = 2*c/5.303
    return center_kp
