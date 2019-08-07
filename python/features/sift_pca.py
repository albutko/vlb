"""
SIFT PCA Implementation based on implementation from
https://github.com/ahojnnes/local-feature-evaluation

Author: Alex Butenko
"""
from .DetectorDescriptorTemplate import DetectorAndDescriptor
import features.feature_utils as fu
import cv2
import numpy as np

from scipy.io import loadmat
import os

MAX_CV_KPTS = 1000
dirname = os.path.dirname(os.path.realpath(__file__))
class sift_pca(DetectorAndDescriptor):
    def __init__(self, eigenmat='./sift_pca_misc/sift-pca.mat'):
        super(
            sift_pca,
            self).__init__(
                name='sift_pca',
                is_detector=True,
                is_descriptor=True,
                is_both=True)

        self.eigvecs = loadmat(os.path.join(dirname,eigenmat)).get('pca_sift_eigvecs')

    def detect_feature(self, image):
        sift = cv2.xfeatures2d.SIFT_create()
        features =  sift.detect(image, None)
        features_pt = np.array([f.pt for f in features])
        responses = np.array([f.response for f in features_pt])

        nb_kpts = MAX_CV_KPTS
        if features_pt.shape[0] < MAX_CV_KPTS:
            nb_kpts = features_pt.shape[0]

        order = responses.argsort()[::-1][:nb_kpts]
        features = features_pt[order,:]
        pts = np.array([features[idx].pt for idx in range(len(features))])
        return pts

    def extract_descriptor(self, image, feature):
        sift = cv2.xfeatures2d.SIFT_create()
        features, full_descriptors =  sift.detectAndCompute(image, feature)
        responses = np.array([f.response for f in features])

        nb_kpts = MAX_CV_KPTS
        if responses.shape[0] < MAX_CV_KPTS:
            nb_kpts = responses.shape[0]

        order = responses.argsort()[::-1][:nb_kpts]
        full_descriptors = full_descriptors[order,:]
        proj_descriptors = np.matmul(self.eigvecs, full_descriptors.T)
        proj_descriptors = proj_descriptors[:80,:].T

        return proj_descriptors

    def extract_all(self, image):
        sift = cv2.xfeatures2d.SIFT_create()
        features, full_descriptors =  sift.detectAndCompute(image, None)
        features_pt = np.array([f.pt for f in features])
        responses = np.array([f.response for f in features])

        nb_kpts = MAX_CV_KPTS
        if features_pt.shape[0] < MAX_CV_KPTS:
            nb_kpts = features_pt.shape[0]

        order = responses.argsort()[::-1][:nb_kpts]

        features = features_pt[order,:]
        full_descriptors = full_descriptors[order,:]
        proj_descriptors = np.matmul(self.eigvecs,full_descriptors.T)
        proj_descriptors = proj_descriptors[:80,:].T

        return (features, proj_descriptors)
