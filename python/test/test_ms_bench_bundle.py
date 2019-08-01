#!/usr/bin/python
# -*- coding: utf-8 -*-
# ===========================================================
#  File Name: test_ms_bench.py
#  Author: Xu Zhang, Columbia University
#  Creation Date: 01-25-2019
#  Last Modified: Mon Apr 15 14:57:08 2019
#
#  Usage: python test_ms_bench.py
#  Description:test matching score benchmark
#
#  Copyright (C) 2018 Xu Zhang
#  All rights reserved.
#
#  This file is made available under
#  the terms of the BSD license (see the COPYING file).
# ===========================================================

import os
import sys
cwd = os.getcwd()
sys.path.insert(0, '{}/python/'.format(cwd))

import bench.Utils
import bench.MatchingScoreBench
import bench.repBench
import bench.repBench
from features.SiftDetectorDescriptorBundle import SiftDetectorDescriptorBundle
from features.spreadout_plus_hardnet import spreadout_plus_hardnet
from features.deepdesc import DeepDesc
from features.cv_convopt import cv_convopt
from features.tfeat import tfeat
import dset.vgg_dataset
import pickle as pkl

import pdb

if __name__ == "__main__":

    # Define matching score benchmark
    ms_bench = bench.MatchingScoreBench.MatchingScoreBench(matchGeometry=False)
    # ms_bench = bench.repBench.repBench()
    ms_result = list()
    # Define dataset
    hp = dset.hpatches_dataset.HPatches_Dataset()
    so = spreadout_plus_hardnet(); so_bundle = SiftDetectorDescriptorBundle(so)
    tf = tfeat(); tf_bundle = SiftDetectorDescriptorBundle(tf)
    dd = DeepDesc(); dd_bundle = SiftDetectorDescriptorBundle(dd)
    co = cv_convopt(); co_bundle = SiftDetectorDescriptorBundle(co)

    # for (modelName, model) in models_to_test:

    so_result = ms_bench.evaluate(hp, so_bundle, use_cache=False, save_result=True)
    tf_result = ms_bench.evaluate(hp, tf_bundle, use_cache=False, save_result=True)
    dd_result = ms_bench.evaluate(hp, dd_bundle, use_cache=False, save_result=True)
    co_result = ms_bench.evaluate(hp, co_bundle, use_cache=False, save_result=True)


    ms_result.append(ms_result)

    ms_result = [so_result,tf_result,dd_result,co_result]

    with open('results.pkl', 'wb') as f:
        pkl.dump(ms_result, f)

    # Show the result
    # for result_term in ms_result[0]['result_term_list']:
    #     bench.Utils.print_result(ms_result, result_term)
    #     bench.Utils.save_result(ms_result, result_term)
    #
    # #show result for different sequences
    # for sequence in vggh.sequence_name_list:
    #     for result_term in ms_result[0]['result_term_list']:
    #         bench.Utils.print_sequence_result(ms_result, sequence, result_term)
    #         bench.Utils.save_sequence_result(ms_result, sequence, result_term)
