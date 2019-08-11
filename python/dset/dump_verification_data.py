"""
file adapted from dump_data.py and config.py in
https://github.com/vcg-uvic/learned-correspondence-release
"""


from __future__ import print_function

import argparse

import itertools
import multiprocessing as mp
import os
import pickle
import sys
import time
import h5py

import urllib
import tarfile

import numpy as np

import cv2
from six.moves import xrange

import os
dirname = os.path.dirname(__file__)

arg_lists = []
parser = argparse.ArgumentParser()
if sys.version_info[0] >= 3:
    from urllib.request import urlretrieve
else:
    from urllib import urlretrieve

def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

def str2bool(v):
    return v.lower() in ("true", "1")



# -----------------------------------------------------------------------------
# Network
net_arg = add_argument_group("Network")
net_arg.add_argument(
    "--net_depth", type=int, default=12, help=""
    "number of layers")
net_arg.add_argument(
    "--net_nchannel", type=int, default=128, help=""
    "number of channels in a layer")
net_arg.add_argument(
    "--net_act_pos", type=str, default="post",
    choices=["pre", "mid", "post"], help=""
    "where the activation should be in case of resnet")
net_arg.add_argument(
    "--net_gcnorm", type=str2bool, default=True, help=""
    "whether to use context normalization for each layer")
net_arg.add_argument(
    "--net_batchnorm", type=str2bool, default=True, help=""
    "whether to use batch normalization")
net_arg.add_argument(
    "--net_bn_test_is_training", type=str2bool, default=False, help=""
    "is_training value for testing")

# -----------------------------------------------------------------------------
# Data
data_arg = add_argument_group("Data")
data_arg.add_argument(
    "--data_dump_prefix", type=str, default=os.path.join(dirname,'../../datasets/Verification/data_dump'), help=""
    "prefix for the dump folder locations")
data_arg.add_argument(
    "--data_tr", type=str, default="reichstag", help=""
    "name of the dataset for train")
data_arg.add_argument(
    "--data_va", type=str, default="reichstag", help=""
    "name of the dataset for valid")
data_arg.add_argument(
    "--data_te", type=str, default="reichstag", help=""
    "name of the dataset for test")
data_arg.add_argument(
    "--data_crop_center", type=str2bool, default=False, help=""
    "whether to crop center of the image "
    "to match the expected input for methods that expect a square input")
data_arg.add_argument(
    "--use_lift", type=str2bool, default=False, help=""
    "if this is set to true, we expect lift to be dumped already for all "
    "images.")


# -----------------------------------------------------------------------------
# Objective
obj_arg = add_argument_group("obj")
obj_arg.add_argument(
    "--obj_num_kp", type=int, default=2000, help=""
    "number of keypoints per image")
obj_arg.add_argument(
    "--obj_top_k", type=int, default=-1, help=""
    "number of keypoints above the threshold to use for "
    "essential matrix estimation. put -1 to use all. ")
obj_arg.add_argument(
    "--obj_num_nn", type=int, default=1, help=""
    "number of nearest neighbors in terms of descriptor "
    "distance that are considered when generating the "
    "distance matrix")
obj_arg.add_argument(
    "--obj_geod_type", type=str, default="episym",
    choices=["sampson", "episqr", "episym"], help=""
    "type of geodesic distance")
obj_arg.add_argument(
    "--obj_geod_th", type=float, default=1e-4, help=""
    "theshold for the good geodesic distance")

# -----------------------------------------------------------------------------
# Loss
loss_arg = add_argument_group("loss")
loss_arg.add_argument(
    "--loss_decay", type=float, default=0.0, help=""
    "l2 decay")
loss_arg.add_argument(
    "--loss_classif", type=float, default=1.0, help=""
    "weight of the classification loss")
loss_arg.add_argument(
    "--loss_essential", type=float, default=0.1, help=""
    "weight of the essential loss")
loss_arg.add_argument(
    "--loss_essential_init_iter", type=int, default=20000, help=""
    "initial iterations to run only the classification loss")

# -----------------------------------------------------------------------------
# Training
train_arg = add_argument_group("Train")
train_arg.add_argument(
    "--run_mode", type=str, default="train", help=""
    "run_mode")
train_arg.add_argument(
    "--train_batch_size", type=int, default=32, help=""
    "batch size")
train_arg.add_argument(
    "--train_max_tr_sample", type=int, default=10000, help=""
    "number of max training samples")
train_arg.add_argument(
    "--train_max_va_sample", type=int, default=1000, help=""
    "number of max validation samples")
train_arg.add_argument(
    "--train_max_te_sample", type=int, default=1000, help=""
    "number of max test samples")
train_arg.add_argument(
    "--train_lr", type=float, default=1e-3, help=""
    "learning rate")
train_arg.add_argument(
    "--train_iter", type=int, default=500000, help=""
    "training iterations to perform")
train_arg.add_argument(
    "--res_dir", type=str, default="./logs", help=""
    "base directory for results")
train_arg.add_argument(
    "--log_dir", type=str, default="", help=""
    "save directory name inside results")
train_arg.add_argument(
    "--test_log_dir", type=str, default="", help=""
    "which directory to test inside results")
train_arg.add_argument(
    "--val_intv", type=int, default=5000, help=""
    "validation interval")
train_arg.add_argument(
    "--report_intv", type=int, default=1000, help=""
    "summary interval")

# -----------------------------------------------------------------------------
# Visualization
vis_arg = add_argument_group('Visualization')
vis_arg.add_argument(
    "--vis_dump", type=str2bool, default=False, help=""
    "turn this on to dump data for visualization"
)
vis_arg.add_argument(
    "--tqdm_width", type=int, default=79, help=""
    "width of the tqdm bar"
)


def writeh5(dict_to_dump, h5node):
    ''' Recursive function to write dictionary to h5 nodes '''

    for _key in dict_to_dump.keys():
        if isinstance(dict_to_dump[_key], dict):
            h5node.create_group(_key)
            cur_grp = h5node[_key]
            writeh5(dict_to_dump[_key], cur_grp)
        else:
            h5node[_key] = dict_to_dump[_key]

def loadh5(dump_file_full_name):
    ''' Loads a h5 file as dictionary '''

    try:
        with h5py.File(dump_file_full_name, 'r') as h5file:
            dict_from_file = readh5(h5file)
    except Exception as e:
        print("Error while loading {}".format(dump_file_full_name))
        raise e

    return dict_from_file

def readh5(h5node):
    ''' Recursive function to read h5 nodes as dictionary '''

    dict_from_file = {}
    for _key in h5node.keys():
        if isinstance(h5node[_key], h5py._hl.group.Group):
            dict_from_file[_key] = readh5(h5node[_key])
        else:
            dict_from_file[_key] = h5node[_key].value

    return dict_from_file

def saveh5(dict_to_dump, dump_file_full_name):
    ''' Saves a dictionary as h5 file '''

    with h5py.File(dump_file_full_name, 'w') as h5file:
        if isinstance(dict_to_dump, list):
            for i, d in enumerate(dict_to_dump):
                newdict = {'dict' + str(i): d}
                writeh5(newdict, h5file)
        else:
            writeh5(dict_to_dump, h5file)

def np_skew_symmetric(v):

    zero = np.zeros_like(v[:, 0])

    M = np.stack([
        zero, -v[:, 2], v[:, 1],
        v[:, 2], zero, -v[:, 0],
        -v[:, 1], v[:, 0], zero,
    ], axis=1)

    return M

def get_episqr(x1, x2, dR, dt):

    num_pts = len(x1)

    # Make homogeneous coordinates
    x1 = np.concatenate([
        x1, np.ones((num_pts, 1))
    ], axis=-1).reshape(-1, 3, 1)
    x2 = np.concatenate([
        x2, np.ones((num_pts, 1))
    ], axis=-1).reshape(-1, 3, 1)

    # Compute Fundamental matrix
    dR = dR.reshape(1, 3, 3)
    dt = dt.reshape(1, 3)
    F = np.repeat(np.matmul(
        np.reshape(np_skew_symmetric(dt), (-1, 3, 3)),
        dR
    ).reshape(-1, 3, 3), num_pts, axis=0)

    x2Fx1 = np.matmul(x2.transpose(0, 2, 1), np.matmul(F, x1)).flatten()

    ys = x2Fx1**2

    return ys.flatten()

def get_episym(x1, x2, dR, dt):

    num_pts = len(x1)

    # Make homogeneous coordinates
    x1 = np.concatenate([
        x1, np.ones((num_pts, 1))
    ], axis=-1).reshape(-1, 3, 1)
    x2 = np.concatenate([
        x2, np.ones((num_pts, 1))
    ], axis=-1).reshape(-1, 3, 1)

    # Compute Fundamental matrix
    dR = dR.reshape(1, 3, 3)
    dt = dt.reshape(1, 3)
    F = np.repeat(np.matmul(
        np.reshape(np_skew_symmetric(dt), (-1, 3, 3)),
        dR
    ).reshape(-1, 3, 3), num_pts, axis=0)

    x2Fx1 = np.matmul(x2.transpose(0, 2, 1), np.matmul(F, x1)).flatten()
    Fx1 = np.matmul(F, x1).reshape(-1, 3)
    Ftx2 = np.matmul(F.transpose(0, 2, 1), x2).reshape(-1, 3)

    ys = x2Fx1**2 * (
        1.0 / (Fx1[..., 0]**2 + Fx1[..., 1]**2) +
        1.0 / (Ftx2[..., 0]**2 + Ftx2[..., 1]**2))

    return ys.flatten()

def get_sampsons(x1, x2, dR, dt):

    num_pts = len(x1)

    # Make homogeneous coordinates
    x1 = np.concatenate([
        x1, np.ones((num_pts, 1))
    ], axis=-1).reshape(-1, 3, 1)
    x2 = np.concatenate([
        x2, np.ones((num_pts, 1))
    ], axis=-1).reshape(-1, 3, 1)

    # Compute Fundamental matrix
    dR = dR.reshape(1, 3, 3)
    dt = dt.reshape(1, 3)
    F = np.repeat(np.matmul(
        np.reshape(np_skew_symmetric(dt), (-1, 3, 3)),
        dR
    ).reshape(-1, 3, 3), num_pts, axis=0)

    x2Fx1 = np.matmul(x2.transpose(0, 2, 1), np.matmul(F, x1)).flatten()
    Fx1 = np.matmul(F, x1).reshape(-1, 3)
    Ftx2 = np.matmul(F.transpose(0, 2, 1), x2).reshape(-1, 3)

    ys = x2Fx1**2 / (
        Fx1[..., 0]**2 + Fx1[..., 1]**2 + Ftx2[..., 0]**2 + Ftx2[..., 1]**2
    )

    return ys.flatten()

def quaternion_from_matrix(matrix, isprecise=False):
    """Return quaternion from rotation matrix.
    If isprecise is True, the input matrix is assumed to be a precise rotation
    matrix and a faster algorithm is used.
    >>> q = quaternion_from_matrix(np.identity(4), True)
    >>> np.allclose(q, [1, 0, 0, 0])
    True
    >>> q = quaternion_from_matrix(np.diag([1, -1, -1, 1]))
    >>> np.allclose(q, [0, 1, 0, 0]) or np.allclose(q, [0, -1, 0, 0])
    True
    >>> R = rotation_matrix(0.123, (1, 2, 3))
    >>> q = quaternion_from_matrix(R, True)
    >>> np.allclose(q, [0.9981095, 0.0164262, 0.0328524, 0.0492786])
    True
    >>> R = [[-0.545, 0.797, 0.260, 0], [0.733, 0.603, -0.313, 0],
    ...      [-0.407, 0.021, -0.913, 0], [0, 0, 0, 1]]
    >>> q = quaternion_from_matrix(R)
    >>> np.allclose(q, [0.19069, 0.43736, 0.87485, -0.083611])
    True
    >>> R = [[0.395, 0.362, 0.843, 0], [-0.626, 0.796, -0.056, 0],
    ...      [-0.677, -0.498, 0.529, 0], [0, 0, 0, 1]]
    >>> q = quaternion_from_matrix(R)
    >>> np.allclose(q, [0.82336615, -0.13610694, 0.46344705, -0.29792603])
    True
    >>> R = random_rotation_matrix()
    >>> q = quaternion_from_matrix(R)
    >>> is_same_transform(R, quaternion_matrix(q))
    True
    >>> R = euler_matrix(0.0, 0.0, np.pi/2.0)
    >>> np.allclose(quaternion_from_matrix(R, isprecise=False),
    ...                quaternion_from_matrix(R, isprecise=True))
    True
    """
    M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
    if isprecise:
        q = np.empty((4, ))
        t = np.trace(M)
        if t > M[3, 3]:
            q[0] = t
            q[3] = M[1, 0] - M[0, 1]
            q[2] = M[0, 2] - M[2, 0]
            q[1] = M[2, 1] - M[1, 2]
        else:
            i, j, k = 1, 2, 3
            if M[1, 1] > M[0, 0]:
                i, j, k = 2, 3, 1
            if M[2, 2] > M[i, i]:
                i, j, k = 3, 1, 2
            t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
            q[i] = t
            q[j] = M[i, j] + M[j, i]
            q[k] = M[k, i] + M[i, k]
            q[3] = M[k, j] - M[j, k]
        q *= 0.5 / math.sqrt(t * M[3, 3])
    else:
        m00 = M[0, 0]
        m01 = M[0, 1]
        m02 = M[0, 2]
        m10 = M[1, 0]
        m11 = M[1, 1]
        m12 = M[1, 2]
        m20 = M[2, 0]
        m21 = M[2, 1]
        m22 = M[2, 2]
        # symmetric matrix K
        K = np.array([[m00-m11-m22, 0.0,         0.0,         0.0],
                         [m01+m10,     m11-m00-m22, 0.0,         0.0],
                         [m02+m20,     m12+m21,     m22-m00-m11, 0.0],
                         [m21-m12,     m02-m20,     m10-m01,     m00+m11+m22]])
        K /= 3.0
        # quaternion is eigenvector of K that corresponds to largest eigenvalue
        w, V = np.linalg.eigh(K)
        q = V[[3, 0, 1, 2], np.argmax(w)]
    if q[0] < 0.0:
        np.negative(q, q)
    return q

def load_geom(geom_file, geom_type, scale_factor, flip_R=False):
    if geom_type == "calibration":
        # load geometry file
        geom_dict = loadh5(geom_file)

        # Check if principal point is at the center
        K = geom_dict["K"]
        # assert(abs(K[0, 2]) < 1e-3 and abs(K[1, 2]) < 1e-3)
        # Rescale calbration according to previous resizing
        S = np.asarray([[scale_factor, 0, 0],
                        [0, scale_factor, 0],
                        [0, 0, 1]])
        K = np.dot(S, K)
        geom_dict["K"] = K
        # Transpose Rotation Matrix if needed
        if flip_R:
            R = geom_dict["R"].T.copy()
            geom_dict["R"] = R
        # append things to list
        geom_list = []
        geom_info_name_list = ["K", "R", "T", "imsize"]
        for geom_info_name in geom_info_name_list:
            geom_list += [geom_dict[geom_info_name].flatten()]
        # Finally do K_inv since inverting K is tricky with theano
        geom_list += [np.linalg.inv(geom_dict["K"]).flatten()]
        # Get the quaternion from Rotation matrices as well
        q = quaternion_from_matrix(geom_dict["R"])
        geom_list += [q.flatten()]
        # Also add the inverse of the quaternion
        q_inv = q.copy()
        np.negative(q_inv[1:], q_inv[1:])
        geom_list += [q_inv.flatten()]
        # Add to list
        geom = np.concatenate(geom_list)


        return geom

def parse_geom(geom, geom_type):

    parsed_geom = {}
    if geom_type == "Homography":
        parsed_geom["h"] = geom.reshape((-1, 3, 3))

    elif geom_type == "Calibration":
        parsed_geom["K"] = geom[:, :9].reshape((-1, 3, 3))
        parsed_geom["R"] = geom[:, 9:18].reshape((-1, 3, 3))
        parsed_geom["t"] = geom[:, 18:21].reshape((-1, 3, 1))
        parsed_geom["K_inv"] = geom[:, 23:32].reshape((-1, 3, 3))
        parsed_geom["q"] = geom[:, 32:36].reshape([-1, 4, 1])
        parsed_geom["q_inv"] = geom[:, 36:40].reshape([-1, 4, 1])

    else:
        raise NotImplementedError(
            "{} is not a supported geometry type!".format(geom_type)
        )

    return parsed_geom


# -----------------------------------------------------------------------------


def download_data():
    print("Downloading Reichstag dataset")
    data_dir = os.path.join(dirname, '../../datasets/Verification/datasets')
    download_url = "http://webhome.cs.uvic.ca/~kyi/files/2018/learned-correspondence/reichstag.tar.gz"
    download_filename = "reichstag.tar.gz"
    data_file = os.path.join(data_dir,download_filename)
    try:
        os.stat(data_dir)
    except:
        os.makedirs(data_dir)


    if os.path.exists(data_dir+"/reichstag") :
        print("already downloaded")
        return

    try:
        urlretrieve(download_url, data_file)
        tar = tarfile.open(data_file)
        tar.extractall(data_dir)
        tar.close()
        os.remove(data_file)
    except:
        print('Cannot download from {}.'.format(download_url))
    print("Download complete")


def setup_dataset(dataset_name):
    """Expands dataset name and directories properly"""

    # Use only the first one for dump
    dataset_name = dataset_name.split(".")[0]

    data_dir = os.path.join(dirname,"../../datasets/Verification/datasets/")


        # Load the data
    data_dir += "reichstag/"
    geom_type = "Calibration"
    vis_th = 100

    return data_dir, geom_type, vis_th

def get_config():
    config, unparsed = parser.parse_known_args()

    # Setup the dataset related things
    for _mode in ["tr", "va", "te"]:
        data_dir, geom_type, vis_th = setup_dataset(
            getattr(config, "data_" + _mode))
        setattr(config, "data_dir_" + _mode, data_dir)
        setattr(config, "data_geom_type_" + _mode, geom_type)
        setattr(config, "data_vis_th_" + _mode, vis_th)

    return config, unparsed


def print_usage():
    parser.print_usage()



eps = 1e-10
use3d = False
config = None

config, unparsed = get_config()



def loadFromDir(train_data_dir, gt_div_str="", bUseColorImage=True,
                input_width=512, crop_center=True, load_lift=False):
    """Loads data from directory.
    train_data_dir : Directory containing data
    gt_div_str : suffix for depth (e.g. -8x8)
    bUseColorImage : whether to use color or gray (default false)
    input_width : input image rescaling size
    """

    # read the list of imgs and the homography
    train_data_dir = train_data_dir.rstrip("/") + "/"
    img_list_file = train_data_dir + "images.txt"
    geom_list_file = train_data_dir + "calibration.txt"
    vis_list_file = train_data_dir + "visibility.txt"
    depth_list_file = train_data_dir + "depth" + gt_div_str + ".txt"
    # parse the file
    image_fullpath_list = []
    with open(img_list_file, "r") as img_list:
        while True:
            # read a single line
            tmp = img_list.readline()
            if type(tmp) != str:
                line2parse = tmp.decode("utf-8")
            else:
                line2parse = tmp
            if not line2parse:
                break
            # strip the newline at the end and add to list with full path
            image_fullpath_list += [train_data_dir +
                                    line2parse.rstrip("\n")]
    # parse the file
    geom_fullpath_list = []
    with open(geom_list_file, "r") as geom_list:
        while True:
            # read a single line
            tmp = geom_list.readline()
            if type(tmp) != str:
                line2parse = tmp.decode("utf-8")
            else:
                line2parse = tmp
            if not line2parse:
                break
            # strip the newline at the end and add to list with full path
            geom_fullpath_list += [train_data_dir +
                                   line2parse.rstrip("\n")]

    # parse the file
    vis_fullpath_list = []
    with open(vis_list_file, "r") as vis_list:
        while True:
            # read a single line
            tmp = vis_list.readline()
            if type(tmp) != str:
                line2parse = tmp.decode("utf-8")
            else:
                line2parse = tmp
            if not line2parse:
                break
            # strip the newline at the end and add to list with full path
            vis_fullpath_list += [train_data_dir + line2parse.rstrip("\n")]

    # parse the file
    if os.path.exists(depth_list_file):
        depth_fullpath_list = []
        with open(depth_list_file, "r") as depth_list:
            while True:
                # read a single line
                tmp = depth_list.readline()
                if type(tmp) != str:
                    line2parse = tmp.decode("utf-8")
                else:
                    line2parse = tmp
                if not line2parse:
                    break
                # strip the newline at the end and add to list with full
                # path
                depth_fullpath_list += [train_data_dir +
                                        line2parse.rstrip("\n")]
    else:
        print("no depth file at {}".format(depth_list_file))
        # import IPython
        # IPython.embed()
        # exit
        depth_fullpath_list = [None] * len(vis_fullpath_list)

    # For each image and geom file in the list, read the image onto
    # memory. We may later on want to simply save it to a hdf5 file
    x = []
    geom = []
    vis = []
    depth = []
    kp = []
    desc = []
    idxImg = 1
    for img_file, geom_file, vis_file, depth_file in zip(
            image_fullpath_list, geom_fullpath_list, vis_fullpath_list,
            depth_fullpath_list):

        print('\r -- Loading Image {} / {}'.format(
            idxImg, len(image_fullpath_list)
        ), end="")
        idxImg += 1

        # ---------------------------------------------------------------------
        # Read the color image
        if not bUseColorImage:
            # If there is not gray image, load the color one and convert to
            # gray
            if os.path.exists(img_file.replace(
                    "image_color", "image_gray"
            )):
                img = cv2.imread(img_file.replace(
                    "image_color", "image_gray"
                ), 0)
                assert len(img.shape) == 2
            else:
                # read the image
                img = cv2.cvtColor(cv2.imread(img_file),
                                   cv2.COLOR_BGR2GRAY)
            if len(img.shape) == 2:
                img = img[..., None]
            in_dim = 1

        else:
            img = cv2.imread(img_file)
            in_dim = 3
        assert(img.shape[-1] == in_dim)

        # Crop center and resize image into something reasonable
        if crop_center:
            rows, cols = img.shape[:2]
            if rows > cols:
                cut = (rows - cols) // 2
                img_cropped = img[cut:cut + cols, :]
            else:
                cut = (cols - rows) // 2
                img_cropped = img[:, cut:cut + rows]
            scale_factor = float(input_width) / float(img_cropped.shape[0])
            img = cv2.resize(img_cropped, (input_width, input_width))
        else:
            scale_factor = 1.0

        # Add to the list
        x += [img.transpose(2, 0, 1)]

        # ---------------------------------------------------------------------
        # Read the geometric information in homography

        geom += [load_geom(
            geom_file,
            "calibration",
            scale_factor,
        )]

        # ---------------------------------------------------------------------
        # Load visibility
        vis += [np.loadtxt(vis_file).flatten().astype("float32")]

        # ---------------------------------------------------------------------
        # Load Depth
        depth += []             # Completely disabled

        if load_lift:
            desc_file = img_file + ".desc.h5"
            with h5py.File(desc_file, "r") as ifp:
                h5_kp = ifp["keypoints"].value[:, :2]
                h5_desc = ifp["descriptors"].value
            # Get K (first 9 numbers of geom)
            K = geom[-1][:9].reshape(3, 3)
            # Get cx, cy
            h, w = x[-1].shape[1:]
            cx = (w - 1.0) * 0.5
            cy = (h - 1.0) * 0.5
            cx += K[0, 2]
            cy += K[1, 2]
            # Get focals
            fx = K[0, 0]
            fy = K[1, 1]
            # New kp
            kp += [
                (h5_kp - np.array([[cx, cy]])) / np.asarray([[fx, fy]])
            ]
            # New desc
            desc += [h5_desc]

    print("")

    return (x, np.asarray(geom),
            np.asarray(vis), depth, kp, desc)

def dump_data_pair(args):
    dump_dir, idx, ii, jj, queue = args

    # queue for monitoring
    if queue is not None:
        queue.put(idx)

    dump_file = os.path.join(
        dump_dir, "idx_sort-{}-{}.h5".format(ii, jj))

    if not os.path.exists(dump_file):
        # Load descriptors for ii
        desc_ii = loadh5(
            os.path.join(dump_dir, "kp-z-desc-{}.h5".format(ii)))["desc"]
        desc_jj = loadh5(
            os.path.join(dump_dir, "kp-z-desc-{}.h5".format(jj)))["desc"]
        # compute decriptor distance matrix
        distmat = np.sqrt(
            np.sum(
                (np.expand_dims(desc_ii, 1) - np.expand_dims(desc_jj, 0))**2,
                axis=2))
        # Choose K best from N
        idx_sort = np.argsort(distmat, axis=1)[:, :config.obj_num_nn]
        idx_sort = (
            np.repeat(
                np.arange(distmat.shape[0])[..., None],
                idx_sort.shape[1], axis=1
            ),
            idx_sort
        )
        distmat = distmat[idx_sort]
        # Dump to disk
        dump_dict = {}
        dump_dict["idx_sort"] = idx_sort
        saveh5(dump_dict, dump_file)


def make_xy(num_sample, pairs, kp, z, desc, img, geom, vis, depth, geom_type,
            cur_folder):

    xs = []
    ys = []
    Rs = []
    ts = []
    img1s = []
    img2s = []
    cx1s = []
    cy1s = []
    f1s = []
    cx2s = []
    cy2s = []
    f2s = []
    k1s = []
    k2s = []

    # Create a random folder in scratch
    dump_dir = os.path.join(cur_folder, "dump")
    if not os.path.exists(dump_dir):
        os.makedirs(dump_dir)

    # randomly suffle the pairs and select num_sample amount
    np.random.seed(1234)
    cur_pairs = [
        pairs[_i] for _i in np.random.permutation(len(pairs))[:num_sample]
    ]
    idx = 0
    for ii, jj in cur_pairs:
        idx += 1
        print(
            "\rExtracting keypoints {} / {}".format(idx, len(cur_pairs)),
            end="")
        sys.stdout.flush()

        # Check and extract keypoints if necessary
        for i in [ii, jj]:
            dump_file = os.path.join(dump_dir, "kp-z-desc-{}.h5".format(i))
            if not os.path.exists(dump_file):
                if kp[i] is None:
                    cv_kp, cv_desc = sift.detectAndCompute(img[i].transpose(
                        1, 2, 0), None)
                    cx = (img[i][0].shape[1] - 1.0) * 0.5
                    cy = (img[i][0].shape[0] - 1.0) * 0.5
                    # Correct coordinates using K
                    cx += parse_geom(geom, geom_type)["K"][i, 0, 2]
                    cy += parse_geom(geom, geom_type)["K"][i, 1, 2]
                    xy = np.array([_kp.pt for _kp in cv_kp])
                    # Correct focals
                    fx = parse_geom(geom, geom_type)["K"][i, 0, 0]
                    fy = parse_geom(geom, geom_type)["K"][i, 1, 1]
                    kp[i] = (
                        xy - np.array([[cx, cy]])
                    ) / np.asarray([[fx, fy]])
                    desc[i] = cv_desc
                if z[i] is None:
                    cx = (img[i][0].shape[1] - 1.0) * 0.5
                    cy = (img[i][0].shape[0] - 1.0) * 0.5
                    fx = parse_geom(geom, geom_type)["K"][i, 0, 0]
                    fy = parse_geom(geom, geom_type)["K"][i, 1, 1]
                    xy = kp[i] * np.asarray([[fx, fy]]) + np.array([[cx, cy]])
                    if len(depth) > 0:
                        z[i] = depth[i][
                            0,
                            np.round(xy[:, 1]).astype(int),
                            np.round(xy[:, 0]).astype(int)][..., None]
                    else:
                        z[i] = np.ones((xy.shape[0], 1))
                # Write descs to harddisk to parallize
                dump_dict = {}
                dump_dict["kp"] = kp[i]
                dump_dict["z"] = z[i]
                dump_dict["desc"] = desc[i]
                saveh5(dump_dict, dump_file)
            else:
                dump_dict = loadh5(dump_file)
                kp[i] = dump_dict["kp"]
                z[i] = dump_dict["z"]
                desc[i] = dump_dict["desc"]
    print("")

    # Create arguments
    pool_arg = []
    idx = 0
    for ii, jj in cur_pairs:
        idx += 1
        pool_arg += [(dump_dir, idx, ii, jj)]
    # Run mp job
    ratio_CPU = 0.8
    number_of_process = int(ratio_CPU * mp.cpu_count())
    pool = mp.Pool(processes=number_of_process)
    manager = mp.Manager()
    queue = manager.Queue()
    for idx_arg in xrange(len(pool_arg)):
        pool_arg[idx_arg] = pool_arg[idx_arg] + (queue,)
    # map async
    pool_res = pool.map_async(dump_data_pair, pool_arg)
    # monitor loop
    while True:
        if pool_res.ready():
            break
        else:
            size = queue.qsize()
            print("\rDistMat {} / {}".format(size, len(pool_arg)), end="")
            sys.stdout.flush()
            time.sleep(1)
    pool.close()
    pool.join()
    print("")
    # Pack data
    idx = 0
    total_num = 0
    good_num = 0
    bad_num = 0
    for ii, jj in cur_pairs:
        idx += 1
        print("\rWorking on {} / {}".format(idx, len(cur_pairs)), end="")
        sys.stdout.flush()

        # ------------------------------
        # Get dR
        R_i = parse_geom(geom, geom_type)["R"][ii]
        R_j = parse_geom(geom, geom_type)["R"][jj]
        dR = np.dot(R_j, R_i.T)
        # Get dt
        t_i = parse_geom(geom, geom_type)["t"][ii].reshape([3, 1])
        t_j = parse_geom(geom, geom_type)["t"][jj].reshape([3, 1])
        dt = t_j - np.dot(dR, t_i)
        # ------------------------------
        # Get sift points for the first image
        x1 = kp[ii]
        y1 = np.concatenate([kp[ii] * z[ii], z[ii]], axis=1)
        # Project the first points into the second image
        y1p = np.matmul(dR[None], y1[..., None]) + dt[None]
        # move back to the canonical plane
        x1p = y1p[:, :2, 0] / y1p[:, 2, 0][..., None]
        # ------------------------------
        # Get sift points for the second image
        x2 = kp[jj]

        x1mat = np.repeat(x1[:, 0][..., None], len(x2), axis=-1)
        y1mat = np.repeat(x1[:, 1][..., None], len(x2), axis=1)
        x1pmat = np.repeat(x1p[:, 0][..., None], len(x2), axis=-1)
        y1pmat = np.repeat(x1p[:, 1][..., None], len(x2), axis=1)
        x2mat = np.repeat(x2[:, 0][None], len(x1), axis=0)
        y2mat = np.repeat(x2[:, 1][None], len(x1), axis=0)
        # Load precomputed nearest neighbors
        idx_sort = loadh5(os.path.join(
            dump_dir, "idx_sort-{}-{}.h5".format(ii, jj)))["idx_sort"]
        # Move back to tuples
        idx_sort = (idx_sort[0], idx_sort[1])
        x1mat = x1mat[idx_sort]
        y1mat = y1mat[idx_sort]
        x1pmat = x1pmat[idx_sort]
        y1pmat = y1pmat[idx_sort]
        x2mat = x2mat[idx_sort]
        y2mat = y2mat[idx_sort]
        # Turn into x1, x1p, x2
        x1 = np.concatenate(
            [x1mat.reshape(-1, 1), y1mat.reshape(-1, 1)], axis=1)
        x1p = np.concatenate(
            [x1pmat.reshape(-1, 1),
             y1pmat.reshape(-1, 1)], axis=1)
        x2 = np.concatenate(
            [x2mat.reshape(-1, 1), y2mat.reshape(-1, 1)], axis=1)

        # make xs in NHWC
        xs += [
            np.concatenate([x1, x2], axis=1).T.reshape(4, 1, -1).transpose(
                (1, 2, 0))
        ]

        # ------------------------------
        # Get the geodesic distance using with x1, x2, dR, dt
        if config.obj_geod_type == "sampson":
            geod_d = get_sampsons(x1, x2, dR, dt)
        elif config.obj_geod_type == "episqr":
            geod_d = get_episqr(x1, x2, dR, dt)
        elif config.obj_geod_type == "episym":
            geod_d = get_episym(x1, x2, dR, dt)
        # Get *rough* reprojection errors. Note that the depth may be noisy. We
        # ended up not using this...
        reproj_d = np.sum((x2 - x1p)**2, axis=1)
        # count inliers and outliers
        total_num += len(geod_d)
        good_num += np.sum((geod_d < config.obj_geod_th))
        bad_num += np.sum((geod_d >= config.obj_geod_th))
        ys += [np.stack([geod_d, reproj_d], axis=1)]
        # Save R, t for evaluation
        Rs += [np.array(dR).reshape(3, 3)]
        ts += [np.array(dt).flatten()]

        # Save img1 and img2 for display
        img1s += [img[ii]]
        img2s += [img[jj]]
        cx = (img[ii][0].shape[1] - 1.0) * 0.5
        cy = (img[ii][0].shape[0] - 1.0) * 0.5
        # Correct coordinates using K
        K1 = parse_geom(geom, geom_type)["K"][ii]
        cx += K1[0, 2]
        cy += K1[1, 2]
        fx = K1[0, 0]
        fy = K1[1, 1]
        if np.isclose(fx, fy):
            f = fx
        else:
            f = (fx, fy)
        cx1s += [cx]
        cy1s += [cy]
        f1s += [f]
        cx = (img[jj][0].shape[1] - 1.0) * 0.5
        cy = (img[jj][0].shape[0] - 1.0) * 0.5
        # Correct coordinates using K
        K2 = parse_geom(geom, geom_type)["K"][jj]
        cx += K2[0, 2]
        cy += K2[1, 2]
        fx = K2[0, 0]
        fy = K2[1, 1]
        if np.isclose(fx, fy):
            f = fx
        else:
            f = (fx, fy)
        cx2s += [cx]
        cy2s += [cy]
        f2s += [f]
        k1s += [K1]
        k2s += [K2]

    # Do *not* convert to np arrays, as the number of keypoints may differ
    # now. Simply return it
    print(".... done")
    if total_num > 0:
        print(" Good pairs = {}, Total pairs = {}, Ratio = {}".format(
            good_num, total_num, float(good_num) / float(total_num)))
        print(" Bad pairs = {}, Total pairs = {}, Ratio = {}".format(
            bad_num, total_num, float(bad_num) / float(total_num)))

    res_dict = {}
    res_dict["xs"] = xs
    res_dict["ys"] = ys
    res_dict["Rs"] = Rs
    res_dict["ts"] = ts
    res_dict["img1s"] = img1s
    res_dict["cx1s"] = cx1s
    res_dict["cy1s"] = cy1s
    res_dict["f1s"] = f1s
    res_dict["img2s"] = img2s
    res_dict["cx2s"] = cx2s
    res_dict["cy2s"] = cy2s
    res_dict["f2s"] = f2s
    res_dict["K1s"] = k1s
    res_dict["K2s"] = k2s

    return res_dict


if __name__ == "__main__":
    print("-------------------------DUMP-------------------------")

    download_data()
    # Read conditions
    crop_center = config.data_crop_center
    data_folder = config.data_dump_prefix
    if config.use_lift:
        data_folder += "_lift"

    # Prepare opencv
    print("Creating Opencv SIFT instance")

    sift = cv2.xfeatures2d.SIFT_create(nfeatures=config.obj_num_kp, contrastThreshold=1e-5)

    # Now start data prep
    print("Preparing data for {}".format(config.data_tr.split(".")[0]))

    # Commented out as this takes a long time to process
    # for _set in ["train", "valid", "test"]:

    # Currently using test set to save time
    for _set in ["test"]:
        num_sample = getattr(
            config, "train_max_{}_sample".format(_set[:2]))

        # Load the data
        print("Loading Raw Data for {}".format(_set))
        if _set == "valid":
            split = "val"
        else:
            split = _set

        img, geom, vis, depth, kp, desc = loadFromDir(
            getattr(config, "data_dir_" + _set[:2]) + split + "/",
            "",
            bUseColorImage=True,
            crop_center=crop_center,
            load_lift=config.use_lift)
        print(kp)
        if len(kp) == 0:
            kp = [None] * len(img)
        if len(desc) == 0:
            desc = [None] * len(img)
        z = [None] * len(img)

        # Generating all possible pairs
        print("Generating list of all possible pairs for {}".format(_set))
        pairs = []
        for ii, jj in itertools.product(xrange(len(img)), xrange(len(img))):
            if ii != jj:
                if vis[ii][jj] > getattr(config, "data_vis_th_" + _set[:2]):
                    pairs.append((ii, jj))
        print("{} pairs generated".format(len(pairs)))

        # Create data dump directory name
        data_names = getattr(config, "data_" + _set[:2])
        data_name = data_names.split(".")[0]
        cur_data_folder = "/".join([
            data_folder,
            data_name,
            "numkp-{}".format(config.obj_num_kp),
            "nn-{}".format(config.obj_num_nn),
        ])
        if not config.data_crop_center:
            cur_data_folder = os.path.join(cur_data_folder, "nocrop")
        if not os.path.exists(cur_data_folder):
            os.makedirs(cur_data_folder)
        suffix = "{}-{}".format(
            _set[:2], getattr(config, "train_max_" + _set[:2] + "_sample"))
        cur_folder = os.path.join(cur_data_folder, suffix)
        if not os.path.exists(cur_folder):
            os.makedirs(cur_folder)

        # Check if we've done this folder already.
        print(" -- Waiting for the data_folder to be ready")
        ready_file = os.path.join(cur_folder, "ready")
        if not os.path.exists(ready_file):
            print(" -- No ready file {}".format(ready_file))
            print(" -- Generating data")
            print(getattr(config, "data_geom_type_" + _set[:2]))
            # Make xy for this pair
            data_dict = make_xy(
                num_sample, pairs, kp, z, desc,
                img, geom, vis, depth, getattr(
                    config, "data_geom_type_" + _set[:2]),
                cur_folder)

            # Let's pickle and save data. Note that I'm saving them
            # individually. This was to have flexibility, but not so much
            # necessary.
            for var_name in data_dict:
                cur_var_name = var_name + "_" + _set[:2]
                out_file_name = os.path.join(cur_folder, cur_var_name) + ".pkl"
                with open(out_file_name, "wb") as ofp:
                    pickle.dump(data_dict[var_name], ofp)

            # Mark ready
            with open(ready_file, "w") as ofp:
                ofp.write("This folder is ready\n")
        else:
            print("Done!")

    #
    # dump_data.py ends here
