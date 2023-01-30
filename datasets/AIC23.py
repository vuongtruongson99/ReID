import glob
import re
import xml.dom.minidom as XD
import os.path as osp
from .base import BaseImageDataset
import logging

logger = logging.getLogger("reid_baseline.train")

class AIC23(BaseImageDataset):
    dataset_dir = "AIC23_Track1_REID"

    def __init__(self, root='../data', verbose=True, crop_test=False, **kwargs):
        super(AIC23, self).__init__()
        self.crop_test = crop_test
        self.dataset_dir = root + '/' + self.dataset_dir
        
        self.train_dir = self.dataset_dir + '/image_train'
        self.query_dir = self.dataset_dir + '/image_query'

        self._check_before_run()

        train = self._process_dir(self.train_dir, relabel=True, query=False)
        query = self._process_dir(self.query_dir, relabel=False, query=True)
        gallery = query

        if verbose:
            print("AIC23 loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = query
        
        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        # self.query = self.train
        # self.gallery = self.train
    
    def _check_before_run(self):
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))

    def _process_dir(self, dir_path, relabel=False, query=False, if_track=False):
        if query:
            xml_dir = self.dataset_dir + '/query_label.xml'
        else:
            xml_dir = self.dataset_dir + '/train_label.xml'

        with open(xml_dir, 'r', encoding='utf-8') as f:
            datasource = f.read()
            f.close()
        info = XD.parseString(datasource).documentElement.getElementsByTagName("Item")

        pid_container = set()
        for element in range(len(info)):
            pid = int(info[element].getAttribute('personID'))
            if pid == -1: continue
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for element in range(len(info)):
            pid, camid = map(int, [info[element].getAttribute('personID'), info[element].getAttribute('cameraID')[1:]])
            image_name = str(info[element].getAttribute("imageName"))
            if pid == -1: continue
            if relabel: pid = pid2label[pid]
            dataset.append((dir_path + '/' + image_name, pid, camid, -1))
        return dataset