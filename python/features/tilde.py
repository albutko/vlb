import os
import subprocess

import cv2
import numpy as np
import features.feature_utils
from features.DetectorDescriptorTemplate import DetectorAndDescriptor

dirname = os.path.dirname(__file__)
class TILDE(DetectorAndDescriptor):
    def __init__(self, temp_img_path="tilde_misc/tempImage.png",
                 temp_kpts_path="tilde_misc/tempKpts.txt",
                 filt_path="tilde_misc/c++/Lib/filters/Mexico24.txt",
                 exe_path="tilde_misc/c++/build/Detector/detect"):
        super(
            TILDE,
            self).__init__(
            name='TILDE',
            is_detector=True,
            is_descriptor=False,
            is_both=False,
            patch_input=False)

        self.temp_img_path = os.path.join(dirname, temp_img_path)
        self.temp_kpts_path = os.path.join(dirname, temp_kpts_path)
        self.filt_path = os.path.join(dirname, filt_path)
        self.exe_path = os.path.join(dirname, exe_path)

        if not os.path.isfile(self.temp_kpts_path):
            with open(self.temp_kpts_path,'w') as f:
                f.write('')

    def detect_feature(self, image):
        cv2.imwrite(self.temp_img_path, image)
        subprocess.run([self.exe_path,
                        self.temp_img_path, self.temp_kpts_path, self.filt_path],
                        shell=False)

        kpts = np.loadtxt(open(self.temp_kpts_path,'r'), dtype='int32', delimiter=',')
        return kpts
