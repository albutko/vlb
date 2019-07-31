import numpy as np

from features.DetectorDescriptorTemplate import DetectorDescriptorBundle
from features.cv_sift import cv_sift

class SiftDetectorDescriptorBundle(DetectorDescriptorBundle):
    def __init__(self, descriptor):
        sift = cv_sift()
        super(SiftDetectorDescriptorBundle, self).__init__(sift, descriptor)
        self.is_detector = True
        self.is_descriptor = True
        self.is_both = True
        self.csv_flag = False
        self.patch_input = True

    def detect_feature(self, image):
        return self.detector.detect_feature_cv_kpt(image)

    def extract_descriptor(self, image, feature):
        return self.descriptor.extract_descriptor(image, feature)

    def extract_all(self, image):
        feature = self.detector.detect_feature_cv_kpt(image)
        descriptor_vector = []
        descriptor_vector = self.descriptor.extract_descriptor(
            image, feature)
        return feature, descriptor_vector
