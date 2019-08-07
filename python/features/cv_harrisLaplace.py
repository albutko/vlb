"""
OpenCV HarrisLaplace Implementation
Author: Alex Butenko
"""
import cv2
import numpy as np
from features.DetectorDescriptorTemplate import DetectorAndDescriptor
import features.feature_utils as fu
import sys

MAX_CV_KPTS = 1000
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
        pts = fu.filter_by_kpt_response(MAX_CV_KPTS, features)
        return pts
