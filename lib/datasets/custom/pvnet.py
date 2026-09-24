import torch.utils.data as data
from pycocotools.coco import COCO
import numpy as np
import os
import cv2
from PIL import Image
from lib.utils.pvnet import pvnet_data_utils, pvnet_linemod_utils, visualize_utils
from lib.utils.linemod import linemod_config
from lib.datasets.augmentation import (
    crop_or_padding_to_fixed_size,
    crop_resize_instance_full,
    crop_resize_instance_v1,
    rotate_instance,
)
import random
import torch
from lib.config import cfg


class Dataset(data.Dataset):

    def __init__(self, ann_file, data_root, split, transforms=None):
        super(Dataset, self).__init__()

        self.data_root = data_root
        self.split = split

        self.coco = COCO(ann_file)
        self.img_ids = np.array(sorted(self.coco.getImgIds()))
        self._transforms = transforms
        self.cfg = cfg
        if self.cfg.train.geometry_mode not in ('legacy_crop', 'full_object'):
            raise ValueError(
                'Unsupported train.geometry_mode: {}'.format(self.cfg.train.geometry_mode)
            )

    def read_data(self, img_id):
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anno = self.coco.loadAnns(ann_ids)[0]

        path = self.coco.loadImgs(int(img_id))[0]['file_name']
        inp = Image.open(path)
        kpt_2d = np.concatenate([anno['fps_2d'], [anno['center_2d']]], axis=0)

        cls_idx = linemod_config.linemod_cls_names.index(anno['cls']) + 1
        mask = pvnet_data_utils.read_linemod_mask(anno['mask_path'], anno['type'], cls_idx)

        return inp, kpt_2d, mask

    def __getitem__(self, index_tuple):
        index, height, width = index_tuple
        img_id = self.img_ids[index]

        img, kpt_2d, mask = self.read_data(img_id)
        if self.split == 'train':
            inp, kpt_2d, mask = self.augment(img, mask, kpt_2d, height, width)
        else:
            inp = img

        if self._transforms is not None:
            inp, kpt_2d, mask = self._transforms(inp, kpt_2d, mask)

        vertex = pvnet_data_utils.compute_vertex(mask, kpt_2d).transpose(2, 0, 1)
        ret = {'inp': inp, 'mask': mask.astype(np.uint8), 'vertex': vertex, 'img_id': img_id, 'meta': {}}
        # visualize_utils.visualize_linemod_ann(torch.tensor(inp), kpt_2d, mask, True)

        return ret

    def __len__(self):
        return len(self.img_ids)

    def augment(self, img, mask, kpt_2d, height, width):
        # add one column to kpt_2d for convenience to calculate
        hcoords = np.concatenate((kpt_2d, np.ones((9, 1))), axis=-1)
        img = np.asarray(img).astype(np.uint8)
        original_img = img.copy()
        original_mask = mask.copy()
        original_hcoords = hcoords.copy()
        foreground = np.sum(mask)
        # randomly mask out to add occlusion
        if foreground > 0:
            if random.random() < self.cfg.train.rotate_rate:
                img, mask, hcoords = rotate_instance(
                    img, mask, hcoords, self.cfg.train.rotate_min, self.cfg.train.rotate_max
                )
            if (
                self.cfg.train.geometry_mode == 'full_object'
                and random.random() < self.cfg.train.full_object_rate
            ):
                img, mask, hcoords = crop_resize_instance_full(
                    img,
                    mask,
                    hcoords,
                    height,
                    width,
                    self.cfg.train.full_object_margin_min,
                    self.cfg.train.full_object_margin_max,
                )
            elif random.random() < self.cfg.train.cropresize_rate:
                img, mask, hcoords = crop_resize_instance_v1(img, mask, hcoords, height, width,
                                                             self.cfg.train.overlap_ratio,
                                                             self.cfg.train.resize_ratio_min,
                                                             self.cfg.train.resize_ratio_max)
            else:
                img, mask = crop_or_padding_to_fixed_size(img, mask, height, width)
        else:
            img, mask = crop_or_padding_to_fixed_size(img, mask, height, width)

        # A crop can occasionally exclude a strongly truncated instance.  An
        # empty mask would make PVNet's vote loss divide by zero, so fall back
        # to a resize of the valid, unaugmented sample in that rare case.
        if not np.any(mask):
            original_height, original_width = original_mask.shape[:2]
            img = cv2.resize(original_img, (width, height), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(
                original_mask, (width, height), interpolation=cv2.INTER_NEAREST
            )
            hcoords = original_hcoords
            hcoords[:, 0] *= width / original_width
            hcoords[:, 1] *= height / original_height

        kpt_2d = hcoords[:, :2]

        return img, kpt_2d, mask
