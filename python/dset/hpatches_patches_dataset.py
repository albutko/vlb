""" HPatches Code adapted from Hpatches Benchmark Github repo
    https://github.com/hpatches/hpatches-benchmark"""

from dset.dataset import SequenceDataset
import urllib
import tarfile
import os
import sys

import cv2
import numpy as np
if sys.version_info[0] >= 3:
    from urllib.request import urlretrieve
else:
    from urllib import urlretrieve


class hpatches_patch_dataset(SequenceDataset):

    def __init__(self,root_dir = './datasets/', download_flag = False, split='a'):
        self.split = split
        super(hpatches_patch_dataset, self).__init__(name='hpatches_patches',
                                                     dataset_info_name ='hpatches_patches_split_{}'.format(self.split),
                                                     root_dir = root_dir, download_flag = download_flag, set_task=False)



    def download(self):
        try:
            os.stat(self.root_dir)
        except:
            os.mkdir(self.root_dir)

        try:
            os.stat('{}{}'.format(self.root_dir, self.name))
            return
        except:
            os.mkdir('{}{}'.format(self.root_dir, self.name))

        download_url = "{}".format(self.url)
        download_filename = "{}/{}.tar.gz".format(self.root_dir, self.name)
        try:
            print("Downloading HPatches Patch dataset from {}".format(download_url))
            urlretrieve(download_url, download_filename)
            tar = tarfile.open(download_filename)
            dd = tar.getnames()[0]
            tar.extractall('{}'.format(self.root_dir))
            tar.close()
            os.remove(download_filename)
            os.rmdir("{}{}".format(self.root_dir, self.name))
            os.rename("{}{}".format(self.root_dir, dd), "{}{}".format(self.root_dir, self.name))
        except Exception as e:
            print(str(e))
            print('Cannot download from {}.'.format(download_url))

    def read_image_data(self):
        """
        Load image data from vggh like dataset
        """
        for i, sequence_name in enumerate(self.sequence_name_list):
            sequence = self.sequences[sequence_name]
            for image_id in sequence.image_id_list:
                img_path = '{}{}/{}'.format(self.root_dir, self.name, sequence.image_dict[image_id].filename)

                sequence.image_dict[image_id].image_path = img_path


    def read_link_data(self):
        self.read_link_data_vggh()


    def get_patch(self, sequence_name, image_id, patch_id):
        """
        Get a sequence of image patches by sequence name and image ID.

        :param sequence_name: Name of the sequence
        :type sequence_name: str
        :param image_id: Image ID
        :type image_id: str
        :returns: image
        :rtype: Image
        """
        image = self.sequences[sequence_name].image_dict[image_id].image_data
        num_patches = self.sequences[sequence_name].image_dict[image_id].num_patches

        return np.split(image, num_patches)[patch_id]

    def get_hpatches_sequence(self, sequence_name):
        """ Get an HPatch sequence using the sequence_name

        :param :param sequence_name: Name of the sequence
        :type sequence_name: str
        :returns hpatches_sequence:
        :rtype: hpatches_sequence """

        return hpatches_sequence(os.path.join(self.root_dir, self.name, sequence_name))

class hpatches_sequence:
    """Class for loading an HPatches sequence from a sequence folder"""

    def __init__(self, base):
        self.itr = ['ref','e1','e2','e3','e4','e5','h1','h2','h3','h4','h5','t1','t2','t3','t4','t5']
        name = base.split('/')
        self.name = name[-1]
        self.base = base
        for t in self.itr:
            im_path = os.path.join(base, t+'.png')
            im = cv2.imread(im_path,0)
            self.N = im.shape[0]/65
            setattr(self, t, np.split(im, self.N))

    def get_center_kp(self, PS=65.):
        c = PS/2.0
        center_kp = cv2.KeyPoint()
        center_kp.pt = (c,c)
        center_kp.size = 2*c/5.303
        return center_kp
