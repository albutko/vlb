"""
OpenCV KAZE Implementation
Author: Alex Butenko
"""
import cv2
import numpy as np
from features.DetectorDescriptorTemplate import DetectorAndDescriptor
import features.feature_utils as fu
import sys

MAX_CV_KPTS = 1000
class cv_kaze(DetectorAndDescriptor):
    def __init__(self):
        super(
            cv_kaze,
            self).__init__(
            name='cv_kaze',
            is_detector=True,
            is_descriptor=True,
            is_both=True,
            patch_input=True)
        self.descriptor = None

    def detect_feature(self, image):
        image = fu.image_resize(image, max_len=640, inter = cv2.INTER_AREA)
        kaze = cv2.KAZE_create()
        features =  kaze.detect(image,None)
        pts = fu.filter_by_kpt_response(MAX_CV_KPTS, features)
        return pts

    def extract_descriptor(self, image, feature):
        image = fu.image_resize(image, max_len=640, inter = cv2.INTER_AREA)
        kaze = cv2.KAZE_create()
        _ , descriptors =  kaze.compute(image, feature)
        return descriptors

    def extract_all(self, image):
        image = fu.image_resize(image, max_len=640, inter = cv2.INTER_AREA)
        kaze = cv2.KAZE_create()
        features, descriptors =  kaze.detectAndCompute(image,None)
        pts, descriptors = fu.filter_by_kpt_response(MAX_CV_KPTS, features, descriptors)
        return (pts, descriptors)

    def extract_descriptor_from_patch(self, patches):
        pass
