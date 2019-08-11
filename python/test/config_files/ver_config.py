import os
import sys
cwd = os.getcwd()
sys.path.insert(0, '{}/python/'.format(cwd))

import features.cyvlsift_official
import features.cv_orb
import features.cv_mser
import features.cv_brisk
import features.cv_fast
import features.cv_akaze
import features.cv_kaze
import features.superpoint
import features.lf_net
import features.transform_covariant
import features.ddet
import features.lift
from importlib import import_module



what_models_to_test = {
    'ransac':{
        'class':'RANSAC',
        'test':True},
    'mlesac':{
        'class':'MLESAC',
        'test':True},
    'lmeds':{
        'class':'LMEDS',
        'test':True},
    'learnedCorres':{
        'class':'learnedCorres',
        'test':True},
    'usac':{
        'class':'USAC',
        'test':True}}


models_to_test = list()

for model, conf in what_models_to_test.items():
    if conf['test']:
        mod = import_module(f"features.{model}")
        cl = getattr(mod, conf['class'])
        models_to_test.append((model, cl()))
