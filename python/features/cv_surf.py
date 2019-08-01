"""
OpenCV SURF Implementation
Author: Alex Butenko
"""
from features.DetectorDescriptorTemplate import DetectorAndDescriptor
import features.feature_utils as fu
import cv2
import numpy as np

MAX_CV_KPTS = 1000
class cv_surf(DetectorAndDescriptor):
    def __init__(self):
        super(
            cv_surf,
            self).__init__(
                name='cv_surf',
                is_detector=True,
                is_descriptor=True,
                is_both=True)


    def detect_feature(self, image):
        image = fu.image_resize(image, max_len=640, inter = cv2.INTER_AREA)
        surf = cv2.xfeatures2d.SURF_create()
        features =  surf.detect(image, None)
        pts = fu.filter_by_kpt_response(MAX_CV_KPTS, features)
        return pts

    def extract_descriptor(self, image, feature):
        image = fu.image_resize(image, max_len=640, inter = cv2.INTER_AREA)
        surf = cv2.xfeatures2d.SURF_create()
        descriptors =  surf.compute(image, feature)
        return descriptors

    def extract_all(self, image):
        image = fu.image_resize(image, max_len=640, inter = cv2.INTER_AREA)
        surf = cv2.xfeatures2d.SURF_create()
        features, descriptors =  surf.detectAndCompute(image, None)
        features, descriptors = fu.filter_by_kpt_response(MAX_CV_KPTS, features, descriptors)
        return (features, descriptors)
