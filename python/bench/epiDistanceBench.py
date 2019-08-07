#!/usr/bin/python
# -*- coding: utf-8 -*-
# ===========================================================
#  File Name: epiDistanceBench.py
#  Author: Alex Butenko, Georiga Institute of Technology
#  Creation Date: 06-01-2019
#  Last Modified: Sat Jun 1 21:46:25 2019
#
#  Description: Epipolar distance benchmark
#
#  Copyright (C) 2019 Alex Butenko
#  All rights reserved.
#
#  This file is made available under
#  the terms of the BSD license (see the COPYING file).
# ===========================================================

"""
This module describe benchmark for repeatability.
"""

import numpy as np
from bench.VerificationBenchmarkTemplate import VerificationBenchmark

import bench.geom as geom

class epiDistanceBench(VerificationBenchmark):
    """
    EpiConstraint Template
    Return repeatability score and number of correspondence
    """
    def __init__(self, tmp_feature_dir='./data/features/',
                 result_dir='./python_scores/'):
        super(epiDistanceBench, self).__init__(name='Epipolar Distance', result_dir=result_dir)
        self.bench_name = 'epiDist'
        self.test_name = 'epiDist'

    def evaluate_unit(self, data_dict):
        """
        Single evaluation unit. Given two sets of points and an estimated Fundamental matrix
        return the Epipolar-Constraint Errors

        :param pts1: points to run from img1
        :type pts1: array
        :param pts2: points to run from img2
        :type pts2: array
        :param task: What to run
        :type task: dict

        See Also
        --------
        evaluate_warpper: How to run the unit.
        dset.dataset.Link: definition of task.

        """
        est_F = data_dict['est_F']
        pts1 = data_dict['px_coords1']
        pts2 = data_dict['px_coords2']

        if est_F is None:
            est_F = geom.get_F_matrix_from_E(data_dict['est_E'],
                                             data_dict['K1'],
                                             data_dict['K2'])
        est_F = data_dict['est_F']
        true_F = data_dict['F']
        pts1 = data_dict['px_coords1']
        pts2 = data_dict['px_coords2']
        true_inliers = data_dict['inlier_mask']

        if est_F is None:
            est_F = geom.get_F_matrix_from_E(data_dict['est_E'],
                                             data_dict['K1'],
                                             data_dict['K2'])

        #Run Processes

        #Get Inliers from estimated Fundamental Matrix
        inlier_pts1, inlier_pts2, inlier_mask  = geom.get_inliers_F(pts1, pts2, est_F, dist_type='symmetric')

        #Epipolar Distance
        epiDists = geom.get_epidist(pts1, pts2, est_F, type='symmetric')
        mean_d = np.mean(epiDists[true_inliers.astype(bool)])
        median_d = np.median(epiDists[true_inliers.astype(bool)])
        if mean_d < 20:
            print(mean_d)


        return mean_d, median_d

    def evaluate(self, dataset, verifier, use_cache=True,
                 save_result=True):
        """
        Main function to call the evaluation wrapper. It could be different for different evaluation

        :param dataset: Dataset to extract the feature
        :type dataset: SequenceDataset
        :param detector: Detector used to extract the feature
        :type detector: DetectorAndDescriptor
        :param use_cache: Load cached feature and result or not
        :type use_cache: boolean
        :param save_result: Save result or not
        :type save_result: boolean
        :param norm_factor: How to normalize the repeatability. Option: minab, a, b
        :type norm_factor: str
        :returns: result
        :rtype: dict

        See Also
        --------
        bench.Benchmark
        bench.Benchmark.evaluate_warpper:
        """
        result = self.evaluate_warpper(dataset, verifier, ['mean_d', 'median_d'],
                                       use_cache=use_cache, save_result=save_result)

        result['bench_name'] = self.bench_name
        return result
