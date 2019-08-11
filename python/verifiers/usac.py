"""USAC implementation"""

import os
import subprocess

import cv2
import numpy as np

from verifiers.VerificationTemplate import VerificationTemplate

dirname = os.path.dirname(__file__)
class USAC(VerificationTemplate):

    def __init__(self):
        super(
            USAC,
            self).__init__(
                name='USAC',
                estimates_essential=False,
                estimates_fundamental=True)

        self.exe_path = os.path.join(dirname, 'usac_misc/build/bin/EstUSAC')
        self.config_path = os.path.join(dirname, 'usac_misc/usac_config.cfg')
        self.corres_path = os.path.join(dirname, 'usac_misc/files/orig_pts.txt')
        self.outpath = os.path.join(dirname, 'usac_misc/files/F.txt')
    def estimate_fundamental_matrix(self, kpts1, kpts2):
        """
        Estimate the Fundamental matrix between 2 images from a set of putative keypoint matches.
        (kpt1s[i], kpts2[i]) are a corresponding matche from image 1 and 2 in camera coordinates

        :param kpts1: Keypoints for image 1
        :type kpts1: np.array (Nx2)
        :param kpts2: Keypoints from image 2
        :type kpts2: np.array (Nx2)
        :returns: E (the 3x3 Essential matrix)
        :rtype: np.array(3x3)
        """
        write_to_points_file(kpts1, kpts2, self.corres_path)

        subprocess.run([self.exe_path, str(0), self.config_path], shell=False)

        F = np.loadtxt(open(self.outpath,'r'))
        F = F.reshape((3,3))

        return F

def write_to_points_file(kpts1, kpts2, file):
    kpts_corres = np.hstack((kpts1, kpts2))
    num_corres = len(kpts_corres)
    np.savetxt(file, kpts_corres, fmt='%.4f', header=str(num_corres), comments='')
