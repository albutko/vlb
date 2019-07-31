"""
OpenCV SIFT Implementation
Author: Alex Butenko
"""
from features.DetectorDescriptorTemplate import DetectorAndDescriptor

import cv2
import numpy as np

MAX_CV_KPTS = 2500
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
        sift = cv2.xfeatures2d.SIFT_create()
        features =  sift.detect(image, None)
        pts = np.array([features[idx].pt for idx in range(len(features))])
        return pts

    def detect_feature_cv_kpt(self, image):
        sift = cv2.xfeatures2d.SIFT_create()
        features =  sift.detect(image, None)
        features_cv = np.array([[f.pt[1], f.pt[0], f.size, f.angle] for f in features])
        responses = np.array([f.response for f in features])

        nb_kpts = MAX_CV_KPTS
        if features_cv.shape[0] < MAX_CV_KPTS:
            nb_kpts = features_cv.shape[0]

        order = responses.argsort()[::-1][:nb_kpts]
        features = features_cv[order,:]

        return features

    def extract_descriptor(self, image, feature):
        sift = cv2.xfeatures2d.SIFT_create()
        descriptors =  sift.compute(image, feature)
        return descriptors

    def extract_all(self, image):
        sift = cv2.xfeatures2d.SIFT_create()
        features, descriptors =  sift.detectAndCompute(image, None)
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
