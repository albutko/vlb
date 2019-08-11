import os

import cv2
import numpy as np
import features.feature_utils
from features.DetectorDescriptorTemplate import DetectorAndDescriptor
import matlab.engine

dirname = os.path.dirname(__file__)

MAX_KPTS = 1000
class DDet(DetectorAndDescriptor):
    def __init__(self, net='detnet_s2.mat', folder="ddet_misc/ddet",vlfeat_path="~/Desktop/vlfeat-0.9.21"):
        super(
            DDet,
            self).__init__(
            name='DDet',
            is_detector=True,
            is_descriptor=False,
            is_both=False,
            patch_input=False)

        self.folder = os.path.join(dirname, folder)
        net_folder = os.path.join(dirname, folder, 'nets')
        matconvnet_folder = os.path.join(dirname, folder, 'matconvnet/matconvnet-master/matlab')
        matconvnet_mex_folder = os.path.join(dirname, folder, 'matconvnet/matconvnet-master/matlab/mex')
        self.eng = matlab.engine.start_matlab()
        self.eng.run(os.path.join(vlfeat_path,'toolbox/vl_setup'), nargout=0)
        self.eng.run(os.path.join(matconvnet_folder,'vl_setupnn'), nargout=0)
        self.eng.addpath(self.folder, nargout=0)
        self.eng.addpath(net_folder, nargout=0)
        self.eng.addpath(matconvnet_folder, nargout=0)
        self.eng.addpath(matconvnet_mex_folder, nargout=0)
        self.temp_img_path=os.path.join(self.folder, 'temp_img.png')
        self.temp_kpts_path=os.path.join(self.folder, 'kpts.txt')
        self.net = net

    def detect_feature(self, image):
        cv2.imwrite(self.temp_img_path, image)
        self.eng.detect_keypoints(self.temp_img_path, self.temp_kpts_path, MAX_KPTS, self.net, nargout=0)
        kpts = np.loadtxt(open(self.temp_kpts_path,'r'), dtype='int32', delimiter=',')

        return kpts
