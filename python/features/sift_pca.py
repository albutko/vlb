"""
SIFT PCA Implementation based on implementation from
https://github.com/ahojnnes/local-feature-evaluation

Author: Alex Butenko
"""
from .DetectorDescriptorTemplate import DetectorAndDescriptor

import cv2
import numpy as np

from scipy.io import loadmat

class sift_pca(DetectorAndDescriptor):
    def __init__(self, eigenmat='./sift_pca_misc/sift-pca.mat'):
        super(
            sift_pca,
            self).__init__(
                name='sift_pca',
                is_detector=True,
                is_descriptor=True,
                is_both=True)

        self.eigvecs = loadmat(eigenmat).get('pca_sift_eigvecs')

    def detect_feature(self, image):
        sift = cv2.xfeatures2d.SIFT_create()
        features =  sift.detect(image, None)
        pts = np.array([features[idx].pt for idx in range(len(features))])
        return pts

    def extract_descriptor(self, image, feature):
        sift = cv2.xfeatures2d.SIFT_create()
        full_descriptors =  sift.compute(image, feature)
        proj_descriptors = self.eigvecs.matmul(full_descriptors.T)
        proj_descriptors = proj_descriptors[:80,:].T

        return proj_descriptors

    def extract_all(self, image):
        sift = cv2.xfeatures2d.SIFT_create()
        features, full_descriptors =  sift.detectAndCompute(image, None)
        proj_descriptors = self.eigvecs.matmul(full_descriptors.T)
        proj_descriptors = proj_descriptors[:80,:].T
        pts = np.array([features[idx].pt for idx in range(len(features))])
        return (pts, proj_descriptors)
