from PIL import Image, ImageFile

from torch.utils.data import Dataset
import os.path as osp
import random
import torch
ImageFile.LOAD_TRUNCATED_IMAGES = True


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img

def read_image_depth(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('L')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class BaseDataset(object):
    """
    Base class of reid dataset
    """

    # def get_imagedata_info(self, data):
    #     pids, cams, tracks = [], [], []
    #
    #     for _, pid, camid, trackid in data:
    #         pids += [pid]
    #         cams += [camid]
    #         tracks += [trackid]
    #     pids = set(pids)
    #     cams = set(cams)
    #     tracks = set(tracks)
    #     num_pids = len(pids)
    #     num_cams = len(cams)
    #     num_imgs = len(data)
    #     num_views = len(tracks)
    #     return num_pids, num_imgs, num_cams, num_views
    def get_imagedata_info(self, data):
        pids, cams, tracks = [], [], []

        for _, _,pid in data:
            pids += [pid]

        pids = set(pids)

        num_pids = len(pids)

        num_imgs = len(data)

        return num_pids, num_imgs



    def print_dataset_statistics(self):
        raise NotImplementedError


class BaseImageDataset(BaseDataset):
    """
    Base class of image reid dataset
    """

    def print_dataset_statistics(self, train, query, gallery):
        num_train_pids, num_train_imgs= self.get_imagedata_info(train)
        num_query_pids, num_query_imgs = self.get_imagedata_info(query)
        num_gallery_pids, num_gallery_imgs= self.get_imagedata_info(gallery)

        print("Dataset statistics:")
        print("  ----------------------------------------")
        print("  subset   | # ids | # images | # cameras")
        print("  ----------------------------------------")
        print("  train    | {:5d} | {:8d} | {:9d}".format(num_train_pids, num_train_imgs, 0))
        print("  query    | {:5d} | {:8d} | {:9d}".format(num_query_pids, num_query_imgs, 0))
        print("  gallery  | {:5d} | {:8d} | {:9d}".format(num_gallery_pids, num_gallery_imgs, 0))
        print("  ----------------------------------------")


class ImageDataset(Dataset):
    def __init__(self, dataset, transform1=None, transform2 = None):
        self.dataset = dataset
        self.transform1 = transform1
        self.transform2 = transform2

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path_color,img_path_depth, pid= self.dataset[index]
        img_color = read_image(img_path_color)
        img_depth = read_image_depth(img_path_depth)

        if self.transform1 is not None:
            img_color = self.transform1(img_color)

        if self.transform2 is not None:
            img_depth = self.transform2(img_depth)

        return img_color,img_depth, pid