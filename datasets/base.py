from PIL import Image, ImageFile
import logging
from torch.utils.data import Dataset
import os.path as osp

ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger('reid_baseline.train')

def read_image(img_path):
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} doesn't exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            logger.info("IOError incurred when reading '{}'. Will redo".format(img_path))
            pass
    return img

class BaseDataset(object):
    def get_imagedata_info(self, data):
        pids, cams, tracks = [], [], []
        for _, pid, camid, trackid in data:
            pids += [pid]
            cams += [camid]
            tracks += [trackid]

        pids = set(pids)
        cams = set(cams)
        tracks = set(tracks)
        num_pids = len(pids)
        num_cams = len(cams)
        num_imgs = len(data)
        return num_pids, num_imgs, num_cams

    def print_dataset_statistics(self):
        raise NotImplementedError

class BaseImageDataset(BaseDataset):
    def print_dataset_statistics(self, train, query, gallery):
        num_train_pids, num_train_imgs, num_train_cams = self.get_imagedata_info(train)
        num_query_pids, num_query_imgs, num_query_cams = self.get_imagedata_info(query)
        num_gallery_pids, num_gallery_imgs, num_gallery_cams = self.get_imagedata_info(gallery)

        logger.info('Dataset statistics:')
        logger.info("  ----------------------------------------")
        logger.info("  subset   | # ids | # images | # cameras")
        logger.info("  ----------------------------------------")
        logger.info("  train    | {:5d} | {:8d} | {:9d}".format(num_train_pids, num_train_imgs, num_train_cams))
        logger.info("  query    | {:5d} | {:8d} | {:9d}".format(num_query_pids, num_query_imgs, num_query_cams))
        logger.info("  gallery  | {:5d} | {:8d} | {:9d}".format(num_gallery_pids, num_gallery_imgs, num_gallery_cams))
        logger.info("  ----------------------------------------")

class ImageDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        img_path, pid, camid, trackid = self.dataset[index]
        img = read_image(img_path)

        if self.transform is not None:
            img = self.transform(img)
        
        return img, pid, camid, trackid, img_path.split('/')[-1]