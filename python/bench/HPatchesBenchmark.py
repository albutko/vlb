#!/usr/bin/python
#-*- coding: utf-8 -*-
#===========================================================
#  File Name: HPatchesBenchmark.py
#  Author: Alex Butenko, Georgia Tech
#  Creation Date: 06/08/2019
#
#  Description: HPatches Benchmark Protocol
#
#  Copyright (C) 2019 Alex Butenko
#  All rights reserved.
#
#  This file is made available under
#  the terms of the BSD license (see the COPYING file).
#===========================================================

"""
This module describes the HPatches benchmark for this process.
A benchmark is given a descriptor and performs the tasks of the HPatches benchmark
"""

import numpy as np
from abc import ABCMeta, abstractmethod
import os
from tqdm import tqdm
import pickle as pkl
import subprocess

import bench.BenchmarkTemplate
from bench.BenchmarkTemplate import Benchmark

class HPatchesBenchmark(Benchmark):
    __metaclass__ = ABCMeta

    """ HPatchesBenchmark

    Attributes
    ----------

    name: str
        Name of the dataset
    tmp_feature_dir: str
        Directory for saving the feature
    result_dir: str
        Directory for saving the final result
    task: list
        list of tasks to evaluate descriptor on
    """

    def __init__(self, tmp_desc_dir='./data/descriptors/',
                 result_dir='./python_scores/', task='matching', split='a'):
        super(HPatchesBenchmark, self).__init__(name='HPatches Benchmark', result_dir='./python_scores/HPatches/')

        self.task = task
        self.split = split
        self.tmp_desc_dir = tmp_desc_dir

    def extract_descriptor_from_patch(self, dataset, detector,
                           use_cache=True):
        """
        Extract feature from HPatches patches.

        :param dataset: Dataset to extract the descriptor
        :type dataset: SequenceDataset
        :param detector: Detector used to extract the descriptor
        :type detector: DetectorAndDescriptor
        :param use_cache: Load cached feature and result or not
        :type use_cache: boolean
        :param save_feature: Save computated feature or not
        :type save_feature: boolean
        :returns: feature, descriptor
        :rtype: dict, dict
        """

        # try:
        #     os.makedirs('{}{}/{}/'.format(self.tmp_desc_dir,
        #                                   dataset.name, detector.name))
        # except BaseException:
        #     pass

        pbar = tqdm(dataset)
        for sequence in pbar:
            try:
                os.makedirs('{}{}/{}/{}/'.format(self.tmp_desc_dir,
                                              dataset.name, detector.name,sequence.name))
            except BaseException:
                pass
            pbar.set_description(
                "Extract feature for {} in {} with {}".format(
                    sequence.name, dataset.name, detector.name))
            for image in sequence.images():
                image = image[1]
                desc_dir = '{}{}'.format(self.tmp_desc_dir, dataset.name)
                descriptor_file_name = '{}{}/{}/{}/{}.csv'.format(self.tmp_desc_dir,
                                                                         dataset.name, detector.name, sequence.name, image.idx)
                if use_cache:
                    try:
                        descriptor = np.load(descriptor_file_name + '.csv')
                        get_feature_flag = True
                    except BaseException:
                        descriptors = extract_descriptors_from_hpatches_image(detector, image.image_data)
                        np.savetxt(descriptor_file_name, descriptors, delimiter=',', fmt='%10.5f')

        return desc_dir


    # Evaluation warpper
    def evaluate_warpper(self, dataset, detector, dist, use_cache=True):
        """
        Load descriptor from cached file. If failed, extract descriptor from image.

        :param dataset: Dataset to extract the feature
        :type dataset: SequenceDataset
        :param detector: Detector used to extract the feature
        :type detector: DetectorAndDescriptor
        :param use_cache: Load cached feature and result or not
        :type use_cache: boolean
        :returns: detector.name, results_dir: detector name and th directory of the results
        :rtype: strs

        See Also
        --------

        detect_feature_custom: Extract feature with customized method (special evaluation).
        extract_descriptor_custom: Extract descriptor with customized (special evaluation).

        """

        desc_dir = self.extract_descriptor_from_patch(dataset, detector, use_cache)
        command = ("python python/bench/hpatches/python/hpatches_eval.py --descr-name={} --task={}"
                        " --descr-dir={} --results-dir={}"
                        " --split={} --dist={}").format(detector.name, self.task,
                                                       desc_dir, self.result_dir,
                                                       self.split, dist)


        subprocess.run(command.split(' '), cwd=os.getcwd(), shell=False)
        return detector.name, self.result_dir

    def evaluate(self, dataset, detector, dist='L2', use_cache=True):
        """
        Main function to run the evaluation wrapper. It could be different for different evaluation

        :param dataset: Dataset to extract the feature
        :type dataset: SequenceDataset
        :param detector: Detector used to extract the feature
        :type detector: DetectorAndDescriptor
        :param dist: Distance name. Valid are {L1,L2}. [default: L2]
        :type dist: string

        See Also
        --------

        evaluate_warpper:
        """
        detector_name, result_dir = self.evaluate_warpper(dataset, detector, dist, use_cache)

        return detector_name, result_dir




def extract_descriptors_from_hpatches_image(detector, hpatch_patch_image):
    """ Take an HPatch patch image and extract descriptors for each patch

    :param detector: Detector to extract the descriptor from patches
    :type detector: SequenceDataset
    :param hpatch_patch_image: Image including multiple patches
    :type np.array: ((num_patches * 65) x 65)
    :returns: descriptors
    :rtype: np.array (num_patches, desc_length)"""

    descriptors = list()
    num_patches = int(hpatch_patch_image.shape[0]//65)
    patch_list = np.split(hpatch_patch_image, num_patches)
    patches = np.zeros((num_patches, 65, 65))

    for i, patch in enumerate(patch_list):
        patches[i, :, :] = patch

    for i in range(num_patches):
        descriptors.append(detector.extract_descriptor_from_patch(patches[i,:,:]))

    descriptors = np.array(descriptors).reshape((num_patches, -1))
    return descriptors
