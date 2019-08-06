

# ===========================================================
#  File Name: pixel_distance.py
#  Author: Alex Butenko, Georgia Institute of Technology
#  Creation Date: 04-25-2019
#
#  This file is made available under
#  the terms of the BSD license (see the COPYING file).
# ===========================================================


from scipy.spatial import distance_matrix
import numpy as np
import cv2

def px_dist_matches(kpts1, kpts2, geo_info, thresh):
    """
        Inputs:
            kpts1: np.array (Nx2) of keypoint coordinates from image 1
            kpts2: np.array (Mx2) of keypoint coordinates from image 2

        Returns:
    """
    homog_1_to_2 = geo_info['H']
    if kpts1.ndim > 2:
        kpts1 = kpts1[:,:2]
        kpts2 = kpts2[:,:2]

    kpts1_clean, kpts2_clean = extract_relevant_keypoints(kpts1, kpts2, geo_info)
    min_kpts = min(kpts1_clean.shape[0], kpts2_clean.shape[0])

    if len(kpts1_clean) == 0 or len(kpts2_clean) == 0:
        return [],[], 0, 0


    kpts1_transformed = transform_points(kpts1_clean, homog_1_to_2)
    kpt_distances = distance_matrix(kpts1_transformed, kpts2_clean)
    match_indices = perform_greedy_matching(kpt_distances, thresh = thresh)
    match_indices = np.array(match_indices)
    kpts1_matched = kpts1_clean[match_indices[:,0],:]
    kpts2_matched = kpts2_clean[match_indices[:,1],:]

    dist = kpt_distances[match_indices[:,0],match_indices[:,1]]

    return  kpts1_matched, kpts2_matched, dist, len(kpts1_clean), len(kpts2_clean)


def extract_relevant_keypoints(kpts1, kpts2, geo_info):
    # Helper Homogeneous Vectors
    img1_h = geo_info['ima_size'][0]
    img1_w = geo_info['ima_size'][1]
    img2_h = geo_info['imb_size'][0]
    img2_w = geo_info['imb_size'][1]

    homog_1_to_2 = geo_info['H']
    homog_2_to_1 = np.linalg.inv(homog_1_to_2)

    kpts1_in2 = transform_points(kpts1, homog_1_to_2)
    kpts2_in1 = transform_points(kpts2, homog_2_to_1)

    indx_kpt1= np.where((kpts1_in2[:,0]<=img2_h) & (kpts1_in2[:,0]>=0) & (kpts1_in2[:,1]<=img2_w) & (kpts1_in2[:,1]>=0))
    indx_kpt2= np.where((kpts2_in1[:,0]<=img1_h) & (kpts2_in1[:,0]>=0) & (kpts2_in1[:,1]<=img1_w) & (kpts2_in1[:,1]>=0))

    return (kpts1[indx_kpt1[0],:], kpts2[indx_kpt2[0],:])

def perform_greedy_matching(kpt_distance_matrix, thresh):
    num_kpt1, num_kpt2 = kpt_distance_matrix.shape

    pair_dists = []
    for i in range(num_kpt1):
        for j in range(num_kpt2):

            pair_dists += [(i,j,kpt_distance_matrix[i,j])]

    pair_dists = np.array(pair_dists)
    inds = np.argsort(pair_dists[:,2])
    pair_dists = pair_dists[inds]
    matches = []
    while pair_dists.size > 0:
        if pair_dists[0,2] > thresh:
            return matches
        a,b = pair_dists[0,:2]
        matches += [(int(a),int(b))]
        pair_dists = pair_dists[1:]
        col0_nondup = np.logical_not(pair_dists[:,0]==a)
        col1_nondup = np.logical_not(pair_dists[:,1]==b)
        non_dup = np.logical_and(col0_nondup,col1_nondup)
        pair_dists = pair_dists[non_dup]

    return matches

def transform_points(kpts, homog):
    """
        Args:
        -   kpts: Numpy n-d array of shape (N,2), representing keypoints detected
                    in an image

        Returns:
        -   kpts_trans: np array of shape (N,2) representing kpts transformed
            by the homograph
    """

    kpts_homogeneous = cv2.convertPointsToHomogeneous(kpts)

    # (N,1,3)->(N,3) because cv2 adds intermediate axis
    kpts_homogeneous = np.squeeze(kpts_homogeneous,axis=1).T

    kpts_homogeneous_transformed = np.matmul(homog, kpts_homogeneous).T
    kpts_transformed = cv2.convertPointsFromHomogeneous(kpts_homogeneous_transformed)

    # (N,1,3)->(N,3) because cv2 has weird axis
    kpts_trans = np.squeeze(kpts_transformed,axis=1)

    return kpts_trans
