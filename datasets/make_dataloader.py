import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from .base import ImageDataset
from .preprocessing import RandomErasing
from .sampler import RandomIdentitySampler
from .AIC23 import AIC23
import logging

__factory = {
    'aic2023': AIC23
}

logger = logging.getLogger("reid_baseline.train")

def train_collate_fn(batch):
    imgs, pids, camids, _, _ = zip(*batch)  # img, pid, camid, trackid, img_path
    pids = torch.tensor(pids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids

def val_collate_fn(batch):
    imgs, pids, camids, trackids, img_paths = zip(*batch)
    return torch.stack(imgs, dim=0), pids, camids, trackids, img_paths


def make_dataloader(cfg):
    if cfg.INPUT.RESIZECROP == True:
        randomcrop = T.RandomResizedCrop(cfg.INPUT.SIZE_TRAN, scale=(0.75, 1.0), ratio=(0.75, 1.3333), interpolation=3)
    else:
        randomcrop = T.RandomCrop(cfg.INPUT.SIZE_TRAIN)
    
    train_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
        T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
        T.Pad(cfg.INPUT.PADDING, padding_mode='constant'),
        randomcrop,
        T.RandomApply([T.ColorJitter(brightness=0.2, contrast=0.15, saturation=0, hue=0)], p=cfg.INPUT.COLORJIT_PROB),
        T.ToTensor(),
        T.RandomErasing(p=cfg.INPUT.RE_PROB, value=cfg.INPUT.PIXEL_MEAN),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
        #RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
    ])

    val_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST, interpolation=3),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])

    num_workers = cfg.DATALOADER.NUM_WORKERS

    dataset = __factory[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR, crop_test=cfg.TEST.CROP_TEST)
    train_set = ImageDataset(dataset.train, train_transforms)
    num_classes = dataset.num_train_pids

    if 'triplet' in cfg.DATALOADER.SAMPLER:
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
            sampler=RandomIdentitySampler(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
            num_workers=num_workers, collate_fn=train_collate_fn
        )
    elif cfg.DATALOADER.SAMPLER == 'softmax':
        logger.info('Using softmax samplert')
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
            collate_fn=train_collate_fn
        )
    else:
        logger.info('Unsupported sampler! Expected softmax or triplet but got {}'.format(cfg.SAMPLER))

    val_set = ImageDataset(dataset.query + dataset.gallery, val_transforms)
    val_loader = DataLoader(
        val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )

    return train_loader, val_loader, len(dataset.query), num_classes