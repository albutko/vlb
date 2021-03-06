"""
Pretrained Outdoot LF-Net Implementation (tensorflow)
Author: Alex Butenko
"""

from features.DetectorDescriptorTemplate import DetectorAndDescriptor
import features.feature_utils as fu

import cv2
import sys
import os
import pickle
import tensorflow as tf
import numpy as np
import importlib
import time
from tqdm import tqdm


dirname = os.path.dirname(__file__)

MODEL_PATH = dirname+'/lf_net_misc/models'
if MODEL_PATH not in sys.path:
    sys.path.append(MODEL_PATH)

sys.path.append(dirname+'/lf_net_misc')
from det_tools import *
from spatial_transformer import transformer_crop

class LFNet(DetectorAndDescriptor):
    def __init__(self):
        super(
            LFNet,
            self).__init__(
                name='LF-Net',
                is_detector=True,
                is_descriptor=True,
                is_both=True,
                patch_input=True)
        #  Initialize network
        tf.reset_default_graph()

        config = pickle.load(open(dirname+'/lf_net_misc/release/models/outdoor/config.pkl','rb'))
        config.max_longer_edge = 640
        config.top_k = 1000
        config.model = dirname+'/lf_net_misc/release/models/outdoor'
        self.photo_ph = tf.placeholder(tf.float32, [1, None, None, 1])
        self.config = config
        self.ops = build_networks(config, self.photo_ph, tf.constant(False))
        print("net build")
        self.sess = self._create_session()
        print("session created")

    def detect_feature(self, image):
        img = fu.all_to_gray(image)
        kpts, _ = self.run(img)

        return kpts

    def extract_descriptor(self, image, feature):
        img = fu.all_to_gray(image)
        _, desc = self.run(img)

        return desc

    def extract_all(self, image):
        img = fu.all_to_gray(image)
        kpts, desc = self.run(img)

        return (kpts, desc)

    def extract_descriptor_from_patch(self, patches):
        pass

    def close_session(self):
        self.sess.close()

    def run(self, img):
        height, width = img.shape[:2]
        longer_edge = max(height, width)
        scale = 1
        if self.config.max_longer_edge > 0 and longer_edge > self.config.max_longer_edge:
            if height > width:
                scale = self.config.max_longer_edge/ height
                new_height = self.config.max_longer_edge
                new_width = int(width * scale)

            else:
                scale = self.config.max_longer_edge/ width
                new_height = int(height * self.config.max_longer_edge / width)
                new_width = self.config.max_longer_edge
            img = cv2.resize(img, (new_width, new_height))
            height, width = img.shape[:2]
        if img.ndim == 3 and img.shape[-1] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = img[None,...,None].astype(np.float32) / 255.0 # normalize 0-1
        assert img.ndim == 4 # [1,H,W,1]

        feed_dict = {
            self.photo_ph: img,
        }

        #Dump keypoint locations and their features
        fetch_dict = {
            'kpts': self.ops['kpts'],
            'feats': self.ops['feats'],
        }

        outs = self.sess.run(fetch_dict, feed_dict=feed_dict)
        return outs['kpts']*1.0/scale, outs['feats']

    def _create_session(self):
        photo_ph = tf.placeholder(tf.float32, [1, None, None, 1]) # input grayscale image, normalized by 0~1
        is_training = tf.constant(False) # Always False in testing

        tfconfig = tf.ConfigProto()
        sess = tf.Session(config=tfconfig)
        sess.run(tf.global_variables_initializer())

        # load model
        saver = tf.train.Saver()
        print('Load trained models...')

        if os.path.isdir(self.config.model):
            checkpoint = tf.train.latest_checkpoint(self.config.model)
            model_dir = self.config.model
        else:
            checkpoint = self.config.model
            model_dir = os.path.dirname(self.config.model)

        if checkpoint is not None:
            print('Checkpoint', os.path.basename(checkpoint))
            print("[{}] Resuming...".format(time.asctime()))
            saver.restore(sess, checkpoint)
        else:
            raise ValueError('Cannot load model from {}'.format(model_dir))

        return sess


def build_networks(config, photo, is_training):
    print("building nets")
    DET = importlib.import_module(config.detector)
    detector = DET.Model(config, is_training)

    if config.input_inst_norm:
        print('Apply instance norm on input photos')
        photos1 = instance_normalization(photo)

    if config.use_nms3d:
        heatmaps, det_endpoints = build_multi_scale_deep_detector_3DNMS(config, detector, photo, reuse=False)
    else:
        heatmaps, det_endpoints = build_multi_scale_deep_detector(config, detector, photo, reuse=False)

    # extract patches
    kpts = det_endpoints['kpts']
    batch_inds = det_endpoints['batch_inds']

    kp_patches = build_patch_extraction(config, det_endpoints, photo)

    # Descriptor
    DESC = importlib.import_module(config.descriptor)
    descriptor = DESC.Model(config, is_training)
    desc_feats, desc_endpoints = descriptor.build_model(kp_patches, reuse=False) # [B*K,D]

    # scale and orientation (extra)
    scale_maps = det_endpoints['scale_maps']
    ori_maps = det_endpoints['ori_maps'] # cos/sin
    degree_maps, _ = get_degree_maps(ori_maps) # degree (rgb psuedo color code)
    kpts_scale = det_endpoints['kpts_scale']
    kpts_ori = det_endpoints['kpts_ori']
    kpts_ori = tf.atan2(kpts_ori[:,1], kpts_ori[:,0]) # radian

    ops = {
        'photo': photo,
        'is_training': is_training,
        'kpts': kpts,
        'feats': desc_feats,
        # EXTRA
        'scale_maps': scale_maps,
        'kpts_scale': kpts_scale,
        'degree_maps': degree_maps,
        'kpts_ori': kpts_ori,
    }

    return ops



def build_deep_detector(config, detector, photos, reuse=False, name='DeepDet'):
    with tf.name_scope(name):
        batch_size = tf.shape(photos)[0]
        height = tf.shape(photos)[1]
        width = tf.shape(photos)[2]

        # Detector
        logits, det_endpoints = detector.build_model(photos, reuse=reuse)
        logits = instance_normalization(logits)
        heatmaps = spatial_softmax(logits, config.sm_ksize, config.com_strength)
        print('PAD_SIZE={}'.format(det_endpoints['pad_size']))
        eof_masks_pad = end_of_frame_masks(height, width, det_endpoints['pad_size'])
        heatmaps = heatmaps * eof_masks_pad

        # Extract Top-K keypoints
        eof_masks_crop = end_of_frame_masks(height, width, config.crop_radius)
        nms_maps = non_max_suppression(heatmaps, config.nms_thresh, config.nms_ksize)
        nms_scores = heatmaps * nms_maps * eof_masks_crop
        top_ks = make_top_k_sparse_tensor(nms_scores, k=config.top_k)
        top_ks = top_ks * nms_maps
        top_ks = tf.stop_gradient(top_ks)

        kpts, batch_inds, num_kpts = extract_keypoints(top_ks)

        if det_endpoints['mso'] == True:
            print('Use multi scale and orientation...')
            scale_log_maps = det_endpoints['scale_maps']
            scale_maps = tf.exp(scale_log_maps)
            ori_maps = det_endpoints['ori_maps']

            kpts_scale = tf.squeeze(batch_gather_keypoints(scale_maps, batch_inds, kpts), axis=1)
            kpts_ori = batch_gather_keypoints(ori_maps, batch_inds, kpts)
        else:
            scale_maps = None
            ori_maps = None
            kpts_scale = None
            kpts_ori = None

        det_endpoints['logits'] = logits
        det_endpoints['top_ks'] = top_ks
        det_endpoints['scale_maps'] = scale_maps
        det_endpoints['ori_maps'] = ori_maps
        det_endpoints['kpts'] = kpts
        det_endpoints['kpts_scale'] = kpts_scale
        det_endpoints['kpts_ori'] = kpts_ori
        det_endpoints['batch_inds'] = batch_inds
        det_endpoints['num_kpts'] = num_kpts

        return heatmaps, det_endpoints

def build_multi_scale_deep_detector(config, detector, photos, reuse=False, name='MSDeepDet'):
    with tf.name_scope(name):

        batch_size = tf.shape(photos)[0]
        height = tf.shape(photos)[1]
        width = tf.shape(photos)[2]

        # Detector
        score_maps_list, det_endpoints = detector.build_model(photos, reuse=reuse)

        scale_factors = det_endpoints['scale_factors']
        scale_factors_tensor = tf.constant(scale_factors, dtype=tf.float32)
        num_scale = len(score_maps_list)

        multi_scale_heatmaps = [None] * num_scale

        for i in range(num_scale):
            logits = instance_normalization(score_maps_list[i])
            _heatmaps = spatial_softmax(logits, config.sm_ksize, config.com_strength)
            _heatmaps = tf.image.resize_images(_heatmaps, (height, width)) # back to original resolution
            multi_scale_heatmaps[i] = _heatmaps
        multi_scale_heatmaps = tf.concat(multi_scale_heatmaps, axis=-1,) # [B,H,W,num_scales]

        if config.soft_scale:
            # max_heatmaps = tf.reduce_max(multi_scale_heatmaps, axis=-1, keep_dims=True) # [B,H,W,1]
            # Maybe softmax have effect of scale-space-NMS
            # softmax_heatmaps = tf.reduce_max(tf.nn.softmax(multi_scale_heatmaps), axis=-1, keep_dims=True)
            # tf.summary.image('softmax_heatmaps', tf.cast(softmax_heatmaps*255, tf.uint8), max_outputs=5)
            max_heatmaps, max_scales = soft_max_and_argmax_1d(multi_scale_heatmaps, axis=-1,
                                                inputs_index=scale_factors_tensor, keep_dims=False,
                                                com_strength1=config.score_com_strength,
                                                com_strength2=config.scale_com_strength) # both output = [B,H,W]
            max_heatmaps = max_heatmaps[..., None] # make max_heatmaps the correct shape
            tf.summary.histogram('max_scales', max_scales)
        else:
            max_heatmaps = tf.reduce_max(multi_scale_heatmaps, axis=-1, keep_dims=True) # [B,H,W,1]
            max_scale_inds = tf.argmax(multi_scale_heatmaps, axis=-1, output_type=tf.int32) # [B,H,W]
            max_scales = tf.gather(scale_factors_tensor, max_scale_inds) # [B,H,W]

        eof_masks_pad = end_of_frame_masks(height, width, det_endpoints['pad_size'])
        max_heatmaps = max_heatmaps * eof_masks_pad

        # Extract Top-K keypoints
        eof_masks_crop = end_of_frame_masks(height, width, config.crop_radius)
        nms_maps = non_max_suppression(max_heatmaps, config.nms_thresh, config.nms_ksize)
        nms_scores = max_heatmaps * nms_maps * eof_masks_crop
        top_ks = make_top_k_sparse_tensor(nms_scores, k=config.top_k)
        top_ks = top_ks * nms_maps
        top_ks = tf.stop_gradient(top_ks)

        ori_maps = det_endpoints['ori_maps']

        kpts, batch_inds, num_kpts = extract_keypoints(top_ks)
        kpts_scale = batch_gather_keypoints(max_scales, batch_inds, kpts)
        kpts_ori = batch_gather_keypoints(ori_maps, batch_inds, kpts)

        if config.soft_kpts:
            # keypoint refinement
            # Use transformer crop to get the patches for refining keypoints to a certain size.
            kp_local_max_scores = transformer_crop(max_heatmaps, config.kp_loc_size, batch_inds, kpts,
                                        kpts_scale=kpts_scale) # omit orientation [N, loc_size, loc_size, 1]
            # Now do a 2d softargmax. I set `do_softmax=True` since the
            # `max_heatmap` is generated by doing softmax
            # individually. However, you might want to see if which works
            # better.
            dxdy = soft_argmax_2d(kp_local_max_scores, config.kp_loc_size, do_softmax=config.do_softmax_kp_refine, com_strength=config.kp_com_strength) # [N,2]
            tf.summary.histogram('dxdy', dxdy)
            # Now add this to the current kpts, so that we can be happy!
            kpts = tf.to_float(kpts) + dxdy * kpts_scale[:, None] * config.kp_loc_size / 2

        det_endpoints['score_maps_list'] = score_maps_list
        det_endpoints['top_ks'] = top_ks
        det_endpoints['kpts'] = kpts # float
        det_endpoints['kpts_scale'] = kpts_scale
        det_endpoints['kpts_ori'] = kpts_ori
        det_endpoints['batch_inds'] = batch_inds
        det_endpoints['num_kpts'] = num_kpts
        det_endpoints['scale_maps'] = max_scales

        det_endpoints['db_max_heatmaps'] = max_heatmaps
        det_endpoints['db_max_scales'] = max_scales
        # det_endpoints['db_max_scales_inds'] = max_scales_inds
        det_endpoints['db_scale_factors_tensor'] = scale_factors_tensor
        # det_endpoints['db_max_heatmaps2'] = max_heatmaps2
        det_endpoints['db_max_heatmaps_org'] = tf.reduce_max(multi_scale_heatmaps, axis=-1, keep_dims=True)
        max_scale_inds = tf.argmax(multi_scale_heatmaps, axis=-1, output_type=tf.int32)
        det_endpoints['db_max_scale_inds'] = max_scale_inds
        det_endpoints['db_max_scales2'] = tf.gather(scale_factors_tensor, max_scale_inds)


        return max_heatmaps, det_endpoints

def build_multi_scale_deep_detector_3DNMS(config, detector, photos, reuse=False, name='MSDeepDet'):
    with tf.name_scope(name):

        batch_size = tf.shape(photos)[0]
        height = tf.shape(photos)[1]
        width = tf.shape(photos)[2]

        # Detector
        score_maps_list, det_endpoints = detector.build_model(photos, reuse=reuse)

        scale_factors = det_endpoints['scale_factors']
        scale_factors_tensor = tf.constant(scale_factors, dtype=tf.float32)
        num_scale = len(score_maps_list)

        scale_logits = [None] * num_scale

        for i in range(num_scale):
            logits = instance_normalization(score_maps_list[i])
            logits = tf.image.resize_images(logits, (height, width)) # back to original resolution
            scale_logits[i] = logits
        scale_logits = tf.concat(scale_logits, axis=-1) # [B,H,W,S]

        # Normalized and Non-max suppressed logits
        scale_heatmaps = soft_nms_3d(scale_logits, ksize=config.sm_ksize, com_strength=config.com_strength)

        if config.soft_scale:
            # max_heatmaps = tf.reduce_max(multi_scale_heatmaps, axis=-1, keep_dims=True) # [B,H,W,1]
            # Maybe softmax have effect of scale-space-NMS
            # softmax_heatmaps = tf.reduce_max(tf.nn.softmax(multi_scale_heatmaps), axis=-1, keep_dims=True)
            # tf.summary.image('softmax_heatmaps', tf.cast(softmax_heatmaps*255, tf.uint8), max_outputs=5)
            max_heatmaps, max_scales = soft_max_and_argmax_1d(scale_heatmaps, axis=-1,
                                                inputs_index=scale_factors_tensor, keep_dims=False,
                                                com_strength1=config.score_com_strength,
                                                com_strength2=config.scale_com_strength) # both output = [B,H,W]
            max_heatmaps = max_heatmaps[..., None] # make max_heatmaps the correct shape
            tf.summary.histogram('max_scales', max_scales)
        else:
            max_heatmaps = tf.reduce_max(scale_heatmaps, axis=-1, keep_dims=True) # [B,H,W,1]
            max_scale_inds = tf.argmax(scale_heatmaps, axis=-1, output_type=tf.int32) # [B,H,W]
            max_scales = tf.gather(scale_factors_tensor, max_scale_inds) # [B,H,W]

        eof_masks_pad = end_of_frame_masks(height, width, det_endpoints['pad_size'])
        max_heatmaps = max_heatmaps * eof_masks_pad

        # Extract Top-K keypoints
        eof_masks_crop = end_of_frame_masks(height, width, config.crop_radius)
        nms_maps = non_max_suppression(max_heatmaps, config.nms_thresh, config.nms_ksize)
        nms_scores = max_heatmaps * nms_maps * eof_masks_crop
        top_ks = make_top_k_sparse_tensor(nms_scores, k=config.top_k)
        top_ks = top_ks * nms_maps
        top_ks = tf.stop_gradient(top_ks)

        ori_maps = det_endpoints['ori_maps']

        kpts, batch_inds, num_kpts = extract_keypoints(top_ks)
        kpts_scale = batch_gather_keypoints(max_scales, batch_inds, kpts)
        kpts_ori = batch_gather_keypoints(ori_maps, batch_inds, kpts)

        if config.soft_kpts:
            # keypoint refinement
            # Use transformer crop to get the patches for refining keypoints to a certain size.
            kp_local_max_scores = transformer_crop(max_heatmaps, config.kp_loc_size, batch_inds, kpts,
                                        kpts_scale=kpts_scale) # omit orientation [N, loc_size, loc_size, 1]
            # Now do a 2d softargmax. I set `do_softmax=True` since the
            # `max_heatmap` is generated by doing softmax
            # individually. However, you might want to see if which works
            # better.
            dxdy = soft_argmax_2d(kp_local_max_scores, config.kp_loc_size, do_softmax=config.do_softmax_kp_refine, com_strength=config.kp_com_strength) # [N,2]
            tf.summary.histogram('dxdy', dxdy)
            # Now add this to the current kpts, so that we can be happy!
            kpts = tf.to_float(kpts) + dxdy * kpts_scale[:, None] * config.kp_loc_size / 2

        det_endpoints['score_maps_list'] = score_maps_list
        det_endpoints['top_ks'] = top_ks
        det_endpoints['kpts'] = kpts # float
        det_endpoints['kpts_scale'] = kpts_scale
        det_endpoints['kpts_ori'] = kpts_ori
        det_endpoints['batch_inds'] = batch_inds
        det_endpoints['num_kpts'] = num_kpts
        det_endpoints['scale_maps'] = max_scales

        return max_heatmaps, det_endpoints

def build_patch_extraction(config, det_endpoints, photos=None, name='PatchExtract'):
    with tf.name_scope(name):
        batch_inds = det_endpoints['batch_inds']
        kpts = det_endpoints['kpts']
        kpts_scale = det_endpoints['kpts_scale']
        kpts_ori = det_endpoints['kpts_ori']

        if config.desc_inputs == 'det_feats':
            feat_maps = tf.identity(det_endpoints['feat_maps'])
        elif config.desc_inputs == 'photos':
            feat_maps = tf.identity(photos)
        elif config.desc_inputs == 'concat':
            feat_maps = tf.concat([photos, det_endpoints['feat_maps']], axis=-1)
        else:
            raise ValueError('Unknown desc_inputs: {}'.format(config.desc_inputs))

        patches = transformer_crop(feat_maps, config.patch_size, batch_inds, kpts,
                        kpts_scale=kpts_scale, kpts_ori=kpts_ori)

        return patches

def build_deep_descriptor(config, descriptor, patches, reuse=False, name='DeepDesc'):
    with tf.name_scope(name):
        desc_feats, desc_endpoints = descriptor.build_model(patches, reuse=reuse) # [B*K,D]
        return desc_feats, desc_endpoints

def build_matching_estimation(config, feats1, feats2, kpts1, kpts2, kpts2w, kpvis2w, dist_thresh=5.0):
    # feats1 = [K1,D] single image
    # feats2 = [K2,D]
    # kpts1 = [K1,2] tf.int32
    # kpts2 = [K2,2] tf.int32
    # kpts2w = [K1,2] tf.float32
    # kpvis2w = [K1,2] take 0 or 1

    # Conver dtype if necessary
    if kpts1.dtype != tf.float32:
        kpts1 = tf.cast(kpts1, tf.float32)
    if kpts2.dtype != tf.float32:
        kpts2 = tf.cast(kpts2, tf.float32)

    nn_dist, nn_inds, _, _, _ = nearest_neighbors(feats1, feats2)
    kpts2_corr = tf.cast(tf.gather(kpts2, nn_inds), tf.float32)
    match_dist = tf.maximum(tf.cast(tf.reduce_sum(tf.squared_difference(kpts2_corr, kpts2w), axis=1), tf.float32), 1e-6)
    match_dist = tf.sqrt(match_dist)
    match_dist_all = match_dist
    is_match = tf.cast(tf.less_equal(match_dist, dist_thresh), tf.float32) * kpvis2w
    num_vis = tf.maximum(tf.reduce_sum(kpvis2w), 1.0)
    match_score = tf.reduce_sum(is_match) / num_vis
    match_dist = tf.reduce_sum(is_match * match_dist) / tf.maximum(tf.reduce_sum(is_match), 1.0)

    match_endpoints = {
        'kpts2_corr': kpts2_corr,
        'is_match': is_match,
        'match_score': match_score,
        'match_dist': match_dist,
        'match_dist_all': match_dist_all,
        'num_vis_kpts': num_vis,
        'num_match': tf.reduce_sum(is_match),
    }

    return match_endpoints

def build_competitor_matching_estimation(config, dist_thresh=5.0):
    # xy_maps1to2 [B,H,W,2], tf.float32
    # support batch_size = 1
    with tf.name_scope('Competitor-matching'):
        feats1_ph = tf.placeholder(tf.float32, [None,None], name='feats1') # [K1,D]
        feats2_ph = tf.placeholder(tf.float32, [None,None], name='feats2') # [K2,D]
        kpts1_ph = tf.placeholder(tf.int32, [None, 2]) # [K1, 2]
        kpts2_ph = tf.placeholder(tf.int32, [None, 2]) # [K2, 2]
        xy_maps1to2_ph = tf.placeholder(tf.float32, [None,None,None,2])
        visible_masks1_ph = tf.placeholder(tf.float32, [None,None,None,1])

        K1 = tf.shape(kpts1_ph)[0]
        batch_inds1 = tf.zeros([K1], dtype=tf.int32)
        kpts2w = batch_gather_keypoints(xy_maps1to2_ph, batch_inds1, kpts1_ph) # float
        kpvis2w = batch_gather_keypoints(visible_masks1_ph, batch_inds1, kpts1_ph)[:,0]

        nn_dist, nn_inds, _, _, _ = nearest_neighbors(feats1_ph, feats2_ph)
        kpts2_corr = tf.cast(tf.gather(kpts2_ph, nn_inds), tf.float32)
        match_dist = tf.maximum(tf.cast(tf.reduce_sum(tf.squared_difference(kpts2_corr, kpts2w), axis=1), tf.float32), 1e-6)
        match_dist = tf.sqrt(match_dist)
        match_dist_all = match_dist
        is_match = tf.cast(tf.less_equal(match_dist, dist_thresh), tf.float32) * kpvis2w
        num_vis = tf.maximum(tf.reduce_sum(kpvis2w), 1.0)
        match_score = tf.reduce_sum(is_match) / num_vis
        match_dist = tf.reduce_sum(is_match * match_dist) / tf.maximum(tf.reduce_sum(is_match), 1.0)
        match_endpoints = {
            'feats1_ph': feats1_ph,
            'feats2_ph': feats2_ph,
            'kpts1_ph': kpts1_ph,
            'kpts2_ph': kpts2_ph,
            'xy_maps1to2_ph': xy_maps1to2_ph,
            'visible_masks1_ph': visible_masks1_ph,
            'kpts2_corr': kpts2_corr,
            'is_match': is_match,
            'match_score': match_score,
            'match_dist': match_dist,
            'match_dist_all': match_dist_all,
            'kpvis2w': kpvis2w,
            'kpts2w': kpts2w,
            'num_vis_kpts': num_vis,
            'num_match': tf.reduce_sum(is_match),
        }

        return match_endpoints
