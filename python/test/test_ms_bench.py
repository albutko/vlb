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
import dset.hpatches_dataset
import pickle as pkl

from config_files.ms_config import models_to_test

if __name__ == "__main__":

    # Define matching score benchmark
    ms_bench = bench.MatchingScoreBench.MatchingScoreBench(matchGeometry=False)

    # Define dataset
    # vggh = dset.vgg_dataset.vggh_Dataset()
    hp = dset.hpatches_dataset.HPatches_Dataset()

    ms_result = list()
    for (modelName, model) in models_to_test:
        vgg_result = ms_bench.evaluate(hp, model, use_cache=False, save_result=True)
    
