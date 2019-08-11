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
    'transform_covariant':  {
        'class':'transform_covariant',
        'test':False},
    'ddet':  {
        'class':'DDet',
        'test':False},
    'cv_sift':  {
        'class':'cv_sift',
        'test':False},
    'cv_harrisLaplace':  {
        'class':'cv_harrisLaplace',
        'test':False},
    'cv_surf':  {
        'class':'cv_surf',
        'test':False},
    'cv_kaze':  {
        'class':'cv_kaze',
        'test':False},
    'lf_net': {
        'class': 'LFNet',
        'test':False},
    'lift': {
        'class': 'LIFT',
        'test':False},
    'd2net': {
        'class': 'd2net',
        'test':False},
    'tilde': {
        'class': 'TILDE',
        'test':False},
    'imips': {
        'class': 'IMIPS',
        'test':True}}


models_to_test = list()

for model, conf in what_models_to_test.items():
    if conf['test']:
        mod = import_module(f"features.{model}")
        cl = getattr(mod, conf['class'])
        models_to_test.append((model, cl()))
