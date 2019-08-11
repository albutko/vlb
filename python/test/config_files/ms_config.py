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
    'superpoint': {
        'class':'SuperPoint',
        'test':False
        },
    'cv_sift':{
        'class':'cv_sift',
        'test':False},
    'sift_pca':{
        'class':'sift_pca',
        'test':False},
    'cv_akaze':{
        'class':'cv_akaze',
        'test':False},
    'cv_kaze':  {
        'class':'cv_kaze',
        'test':False},
    'cv_surf':  {
        'class':'cv_surf',
        'test':False},
    'lf_net':  {
        'class':'LFNet',
        'test':False},
    'd2net':  {
        'class':'d2net',
        'test':True}}

models_to_test = list()

for model, conf in what_models_to_test.items():
    if conf['test']:
        mod = import_module(f"features.{model}")
        cl = getattr(mod, conf['class'])
        models_to_test.append((model, cl()))
