import os
import sys
cwd = os.getcwd()
sys.path.insert(0, '{}/python/'.format(cwd))

import features.cyvlsift_official
from features.SiftDetectorDescriptorBundle import SiftDetectorDescriptorBundle
from importlib import import_module



what_models_to_test = {
    'deepdesc':{
        'class':'DeepDesc',
        'test':False},
    'cv_convopt':{
        'class':'cv_convopt',
        'test':True},
    'tfeat':{
        'class':'tfeat',
        'test':True},
    'spreadout_plus_hardnet': {
        'class':'spreadout_plus_hardnet',
        'test':True
        }}

models_to_test = list()

for model, conf in what_models_to_test.items():
    if conf['test']:
        mod = import_module(f"features.{model}")
        cl = getattr(mod, conf['class'])

        models_to_test.append((model, SiftDetectorDescriptorBundle(cl())))
