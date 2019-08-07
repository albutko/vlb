"""
Pretrained LIFT Implementation (tensorflow)
Author: Alex Butenko
"""

from features.DetectorDescriptorTemplate import DetectorAndDescriptor
import features.feature_utils as fu

import cv2
import sys
import os
import numpy as np

import subprocess

dirname = os.path.dirname(__file__)

sys.path.append(os.path.join(dirname,'lift_misc'))

from features.lift_misc.utils import loadKpListFromTxt


class LIFT(DetectorAndDescriptor):
    def __init__(self, model_folder= os.path.join(dirname,'lift_misc/release-aug'),
                 temp_img_path=os.path.join(dirname,'lift_misc/tempImg.png'),
                 temp_kpts_path=os.path.join(dirname,'lift_misc/tempKpts.txt'),
                 temp_desc_path=os.path.join(dirname,'lift_misc/tempDesc.txt')) :
        super(
            LIFT,
            self).__init__(
                name='LIFT',
                is_detector=True,
                is_descriptor=True,
                is_both=True,
                patch_input=False)
        self.temp_img_path = temp_img_path
        self.temp_kpts_path = temp_kpts_path
        self.temp_desc_path = temp_desc_path
        self.model_folder = model_folder

    def detect_feature(self, image):
        command = self._get_command('kp')
        img = fu.all_to_gray(image)

        cv2.imwrite(self.temp_img_path, img)
        subprocess.run(command,
                        shell=True)

        kpts_raw = loadKpListFromTxt(self.temp_kpts_path)
        kpts = []
        for k in kpts_raw:
            kpts.append([int(k[0]), int(k[1])])

        kpts = np.array(kpts)
        return kpts

    def extract_descriptor(self, image, feature):
        img = fu.all_to_gray(image)
        _, desc = self.run(img)

        return desc

    def extract_all(self, image):
        img = fu.all_to_gray(image)
        kpts, desc = self.run(img)

        return (kpts, desc)


    def _get_command(self, subtask):
        if subtask == 'kp':
            command = "python {} --task=test --subtask=kp --logdir={} --test_img_file={} --test_out_file={}".format(
                            os.path.join(dirname, 'lift_misc/main.py'),
                            self.model_folder, self.temp_img_path, self.temp_kpts_path)
        elif subtask == 'desc':
            command = "python {} --task=test --subtask=desc --logdir={} --test_img_file={}  --test_kp_file={} --test_out_file={}".format(
                            os.path.join(dirname, 'lift_misc/main.py'),
                            self.model_files, self.temp_img_path, self.temp_kpts_path, self.temp_desc_path)
        else:
            command = ''

        return command
