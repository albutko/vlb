import os
import sys
cwd = os.getcwd()
sys.path.insert(0, '{}/python/'.format(cwd))

import bench.Utils
import bench.HPatchesBenchmark
import features.cv_sift
import features.deepdesc
import features.tfeat


import dset.hpatches_patches_dataset


if __name__ == "__main__":
    split = 'small'
    # Define matching score benchmark
    hp_bench = bench.HPatchesBenchmark.HPatchesBenchmark(split=split)
    print("bench loaded")
    # Define features
    # cv_sift = features.cv_sift.cv_sift()
    # tfeat = features.tfeat.tfeat()
    deepdesc = features.deepdesc.DeepDesc()
    # tc = features.transform_covariant.transform_covariant()

    hpatch = dset.hpatches_patches_dataset.hpatches_patch_dataset(split=split)

    # Do the evaluation
    _, _ = hp_bench.evaluate(
        hpatch, deepdesc, use_cache=True, dist='L2')
