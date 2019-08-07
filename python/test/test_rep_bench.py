#!/usr/bin/python
# -*- coding: utf-8 -*-
# ===========================================================
#  File Name: test_rep_bench.py
#  Author: Xu Zhang, Columbia University
#  Creation Date: 01-25-2019
#  Last Modified: Tue Mar  5 00:03:28 2019
#
#  Usage: python test_rep_bench.py
#  Description:test repeatability benchmark
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
import bench.repBench
import dset.hpatches_dataset

from config_files.rep_config import models_to_test

if __name__ == "__main__":

    # Define repeatability benchmark
    rep_bench = bench.repBench.repBench()

    # Define dataset
    hp = dset.hpatches_dataset.HPatches_Dataset()

    for (modelName, model) in models_to_test:
        hp_result = rep_bench.evaluate(hp, model, use_cache=True, save_result=True)
        if hasattr(model, 'close'):
            model.close()
