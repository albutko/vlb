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
from features.lift_misc.utils import loadKpListFromTxt, loadh5


class LIFT(DetectorAndDescriptor):
    def __init__(self, model_folder= os.path.join(dirname,'lift_misc/release-aug'),
                 temp_img_path=os.path.join(dirname,'lift_misc/tempImg.png'),
                 temp_kpts_path=os.path.join(dirname,'lift_misc/tempKpts.txt'),
                 temp_desc_path=os.path.join(dirname,'lift_misc/tempDesc.h5'),
                 temp_ori_path=os.path.join(dirname,'lift_misc/tempOri.txt')) :
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
        self.temp_ori_path = temp_ori_path
        self.model_folder = model_folder

    def detect_feature(self, image):
        commands = self._get_command_list('kp')
        img = fu.all_to_gray(image)

        cv2.imwrite(self.temp_img_path, img)
        for command in commands:
            subprocess.run(command,
                            shell=True)

        kpts_raw = loadKpListFromTxt(self.temp_kpts_path)
        kpts = []
        for k in kpts_raw:
            kpts.append([int(k[0]), int(k[1])])

        kpts = np.array(kpts)
        return kpts

    def extract_descriptor(self, image, feature):
        commands = self._get_command_list('desc')
        img = fu.all_to_gray(image)

        cv2.imwrite(self.temp_img_path, img)
        for command in commands:
            subprocess.run(command,
                            shell=True)

        data_dict = loadh5(self.temp_desc_path)
        desc = data_dict.get('descriptors')

        return desc

    def extract_all(self, image):
        commands = self._get_command_list('desc')
        img = fu.all_to_gray(image)

        cv2.imwrite(self.temp_img_path, img)
        for command in commands:
            subprocess.run(command,
                            shell=True)

        kpts_raw = loadKpListFromTxt(self.temp_kpts_path)
        data_dict = loadh5(self.temp_desc_path)
        desc = data_dict.get('descriptors')
        kpts = []
        for k in kpts_raw:
            kpts.append([int(k[0]), int(k[1])])

        kpts = np.array(kpts)

        return (kpts, desc)


    def _get_command_list(self, subtask):
        command_list = list()
        if subtask == 'kp' or subtask == 'desc':
            command = "python {} --task=test --subtask=kp --logdir={} --test_img_file={} --test_out_file={}".format(
                            os.path.join(dirname, 'lift_misc/main.py'),
                            self.model_folder, self.temp_img_path, self.temp_kpts_path)

            command_list.append(command)

        if subtask == 'desc':
            command = "python {} --task=test --subtask=ori --logdir={} --test_img_file={}  --test_kp_file={} --test_out_file={}".format(
                            os.path.join(dirname, 'lift_misc/main.py'),
                            self.model_folder, self.temp_img_path, self.temp_kpts_path, self.temp_ori_path)

            command_list.append(command)

            command = "python {} --task=test --subtask=desc --logdir={} --test_img_file={}  --test_kp_file={} --test_out_file={}".format(
                            os.path.join(dirname, 'lift_misc/main.py'),
                            self.model_folder, self.temp_img_path, self.temp_ori_path, self.temp_desc_path)

            command_list.append(command)
        else:
            command = ''

        return command_list
