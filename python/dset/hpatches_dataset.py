from dset.dataset import SequenceDataset
import urllib
import tarfile
import os
import sys
import scipy

if sys.version_info[0] >= 3:
    from urllib.request import urlretrieve
else:
    from urllib import urlretrieve


class HPatches_Dataset(SequenceDataset):

    def __init__(self,root_dir = './datasets/', download_flag = False):
        super(HPatches_Dataset,self).__init__(name = 'hpatches_full', root_dir = root_dir, download_flag = download_flag, set_task=True)

    def download(self):
        try:
            os.stat(self.root_dir)
        except:
            os.mkdir(self.root_dir)

        try:
            os.stat('{}{}'.format(self.root_dir,self.name))
        except:
            os.mkdir('{}{}'.format(self.root_dir,self.name))

        download_url = "{}".format(self.url)
        download_filename = "{}/{}.tar.gz".format(self.root_dir, self.name)
        try:
            print("Downloading HPatches from {}".format(download_url))
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

        for sequence_name in self.sequence_name_list:
            sequence = self.sequences[sequence_name]
            for image_id in sequence.image_id_list:
                sequence.image_dict[image_id].image_path = '{}{}/{}'.format(self.root_dir, self.name, sequence.image_dict[image_id].filename)

    def set_task(self):
        """
        Deprecated
        """

        for sequence_name in self.sequence_name_list:
            sequence = self.sequences[sequence_name]
            for link_id in sequence.link_id_list:
                this_link = sequence.link_dict[link_id]
                image_a = sequence.image_dict[this_link.source]
                image_b = sequence.image_dict[this_link.target]
                this_link.task['ima'] = str(image_a.idx)
                this_link.task['imb'] = str(image_b.idx)
                image_a_data = scipy.ndimage.imread(image_a.image_path)
                image_b_data = scipy.ndimage.imread(image_b.image_path)
                try:
                    imga_ch = image_a_data.shape[2]
                except:
                    imga_ch = 1
                try:
                    imgb_ch = image_b_data.shape[2]
                except:
                    imgb_ch = 1

                this_link.task['ima_size'] = [image_a_data.shape[0], image_a_data.shape[1], imga_ch]
                this_link.task['imb_size'] = [image_b_data.shape[0], image_b_data.shape[1], imgb_ch]
                this_link.task['H'] = this_link.transform_matrix

                this_link.task['name'] = str(sequence.name)
                this_link.task['description'] = {}
                this_link.task['description']['impair'] = [str(image_a.idx), str(image_b.idx)]
                try:
                    this_link.task['description']['nuisanceName'] = str(sequence.label)
                    this_link.task['description']['nuisanceValue'] = str(imageb.label)
                except:
                    pass
    def read_link_data(self):
        self.read_link_data_vggh()
