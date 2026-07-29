import os
from collections import defaultdict

import numpy as np
import pycocotools.coco as coco
import torch

from lib.config import cfg
from lib.datasets.dataset_catalog import DatasetCatalog


def _mean(values):
    return float(np.mean(values)) if values else None


def _median(values):
    return float(np.median(values)) if values else None


def _pck_thresholds():
    raw = os.environ.get("PANEL_PVNET_VAL_PCK_THRESHOLDS", "2,5,10")
    thresholds = [float(value.strip()) for value in raw.split(",") if value.strip()]
    if not thresholds:
        raise RuntimeError("PANEL_PVNET_VAL_PCK_THRESHOLDS non contiene soglie")
    return thresholds


class Evaluator:

    def __init__(self, result_dir):
        del result_dir
        args = DatasetCatalog.get(cfg.test.dataset)
        self.coco = coco.COCO(args["ann_file"])
        self.thresholds = _pck_thresholds()
        self.reset()

    def reset(self):
        self.keypoint_errors = []
        self.inside_errors = []
        self.outside_errors = []
        self.per_keypoint_errors = defaultdict(list)
        self.mask_ious = []

    @staticmethod
    def ground_truth_keypoints(annotation):
        return np.concatenate(
            [
                np.asarray(annotation["fps_2d"], dtype=np.float64),
                np.asarray([annotation["center_2d"]], dtype=np.float64),
            ],
            axis=0,
        )

    @staticmethod
    def selected_ids(annotation, count):
        ids = annotation.get("selected_keypoint_ids")
        if ids is None:
            ids = list(range(count))
        ids = [int(value) for value in ids]
        if len(ids) != count:
            raise RuntimeError(
                "selected_keypoint_ids non coerente con il numero di target PVNet"
            )
        return ids

    def evaluate(self, output, batch):
        predicted_batch = output["kpt_2d"].detach().cpu().numpy()
        predicted_masks = torch.argmax(output["seg"], dim=1).detach().cpu().numpy()
        ground_truth_masks = batch["mask"].detach().cpu().numpy()
        image_ids = batch["img_id"].detach().cpu().numpy()

        for batch_index, image_id_value in enumerate(image_ids):
            image_id = int(image_id_value)
            annotation = self.coco.loadAnns(
                self.coco.getAnnIds(imgIds=image_id)
            )[0]
            image_info = self.coco.loadImgs(image_id)[0]
            ground_truth = self.ground_truth_keypoints(annotation)
            predicted = np.asarray(predicted_batch[batch_index], dtype=np.float64)
            if predicted.shape != ground_truth.shape:
                raise RuntimeError(
                    f"Shape keypoint non coerenti per image_id={image_id}: "
                    f"pred={predicted.shape}, gt={ground_truth.shape}"
                )

            errors = np.linalg.norm(predicted - ground_truth, axis=1)
            self.keypoint_errors.extend(errors.tolist())
            ids = self.selected_ids(annotation, len(errors))
            for keypoint_id, error in zip(ids, errors):
                self.per_keypoint_errors[keypoint_id].append(float(error))

            width = int(image_info["width"])
            height = int(image_info["height"])
            inside = (
                (ground_truth[:, 0] >= 0)
                & (ground_truth[:, 0] < width)
                & (ground_truth[:, 1] >= 0)
                & (ground_truth[:, 1] < height)
            )
            self.inside_errors.extend(errors[inside].tolist())
            self.outside_errors.extend(errors[~inside].tolist())

            predicted_mask = predicted_masks[batch_index] == 1
            ground_truth_mask = ground_truth_masks[batch_index] != 0
            intersection = np.logical_and(predicted_mask, ground_truth_mask).sum()
            union = np.logical_or(predicted_mask, ground_truth_mask).sum()
            self.mask_ious.append(float(intersection / union) if union else 1.0)

    def summarize(self):
        result = {
            "kp_mean_px": _mean(self.keypoint_errors),
            "kp_median_px": _median(self.keypoint_errors),
            "mask_iou": _mean(self.mask_ious),
            "mask_ap70": _mean(
                [float(value >= 0.7) for value in self.mask_ious]
            ),
        }
        if self.inside_errors:
            result["kp_inside_mean_px"] = _mean(self.inside_errors)
            result["kp_inside_median_px"] = _median(self.inside_errors)
        if self.outside_errors:
            result["kp_outside_mean_px"] = _mean(self.outside_errors)
            result["kp_outside_median_px"] = _median(self.outside_errors)
        for threshold in self.thresholds:
            name = "{:g}".format(threshold).replace(".", "_")
            result["pck_{}px".format(name)] = _mean(
                [float(error <= threshold) for error in self.keypoint_errors]
            )

        print("Validation 2D PVNet")
        print("  keypoint mean/median [px]: {:.4f} / {:.4f}".format(
            result["kp_mean_px"], result["kp_median_px"]
        ))
        print("  mask IoU / AP70: {:.4f} / {:.4f}".format(
            result["mask_iou"], result["mask_ap70"]
        ))
        for keypoint_id in sorted(self.per_keypoint_errors):
            values = self.per_keypoint_errors[keypoint_id]
            result["kp_{}_mean_px".format(keypoint_id)] = _mean(values)
            result["kp_{}_median_px".format(keypoint_id)] = _median(values)
            print(
                "  keypoint {} mean/median [px]: {:.4f} / {:.4f}".format(
                    keypoint_id, _mean(values), _median(values)
                )
            )

        self.reset()
        return result
