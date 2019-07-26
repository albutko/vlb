"""
OpenCV BRISK Implementation
Author: Alex Butenko
"""
import cv2
import numpy as np
from features.DetectorDescriptorTemplate import DetectorAndDescriptor
import sys

class cv_harrisLaplace(DetectorAndDescriptor):
    def __init__(self):
        super(
            cv_harrisLaplace,
            self).__init__(
            name='cv_harrisLaplace',
            is_detector=True,
            is_descriptor=False,
            is_both=False,
            patch_input=False)


    def detect_feature(self, image):
        harLap = cv2.xfeatures2d.HarrisLaplaceFeatureDetector_create()
        features =  harLap.detect(image,None)
        pts = np.array([features[idx].pt for idx in range(len(features))])
        return pts
