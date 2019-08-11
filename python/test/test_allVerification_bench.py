#!/usr/bin/python
# -*- coding: utf-8 -*-
# ===========================================================
#  File Name: test_allVerification_bench.py
#  Author: Alex Butenko, Georiga Institute of Technology
#  Creation Date: 06-01-2019
#  Last Modified: Sat Jun 1 21:46:25 2019
#
#  Description: Test all verification tasks
#
#  Copyright (C) 2019 Alex Butenko
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
import bench.allVerificationBenchs
import dset.verification_dataset

from config_files.ver_config import models_to_test

if __name__ == "__main__":

    # Define epiConstraintBench benchmark
    ver_bench = bench.allVerificationBenchs.allVerificationBenchs()

    # Define dataset
    dataset = dset.verification_dataset.verification_dataset(['reichstag'])

    ver_result = []
    for (modelName, model) in models_to_test:
        results = ver_bench.evaluate(dataset, model, use_cache=False, save_result=True)
        ver_result.append(results)

    # Show the result
    for result_term in ver_result[0]['result_term_list']:
        bench.Utils.print_result(ver_result, result_term)
        bench.Utils.save_result(ver_result, result_term)
