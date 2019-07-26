import os
import subprocess

import cv2
import numpy as np
import features.feature_utils
from features.DetectorDescriptorTemplate import DetectorAndDescriptor

dirname = os.path.dirname(__file__)
class DeepDesc(DetectorAndDescriptor):
    def __init__(self, temp_img_path="deepdesc_misc/tempImage.png",
                 temp_desc_path="deepdesc_misc/tempDesc.txt",
                 exe_path="deepdesc_misc/desc.lua"):
        super(
            DeepDesc,
            self).__init__(
            name='DeepDesc',
            is_detector=False,
            is_descriptor=True,
            is_both=False,
            patch_input=True,
            can_batch=True)

        self.temp_img_path = os.path.join(dirname, temp_img_path)
        self.temp_desc_path = os.path.join(dirname, temp_desc_path)
        self.exe_path = os.path.join(dirname, exe_path)

        if not os.path.isfile(self.temp_desc_path):
            with open(self.temp_desc_path,'w') as f:
                f.write('')

    def extract_descriptors_from_patch_batch(self, batch):
        batch_size = batch.shape[0]
        batch = np.concatenate(batch, axis = 0)
        cv2.imwrite(self.temp_img_path, batch)
        subprocess.run(['th', self.exe_path,
                        self.temp_img_path, self.temp_desc_path, str(batch_size)],
                        shell=False)

        desc = np.loadtxt(open(self.temp_desc_path,'r'), dtype='float64', delimiter=',')
        return desc
