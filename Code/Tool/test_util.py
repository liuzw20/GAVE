from pathlib import Path
from argparse import Namespace
from typing import Union

import numpy as np
from sklearn.metrics import confusion_matrix
from skimage import io
from skimage.transform import resize
from skimage.morphology import skeletonize
from medpy import metric 
from tqdm import tqdm

from Tool.topo_metric import topo_metric

def dice_coefficient(
    ground_truth,
    prediction,
    mask=None,
    threshold=0.5,
    nan_for_nonexisting=True 
):
    if mask is not None:
        ground_truth = np.where(mask, ground_truth, 0)  
        prediction = np.where(mask, prediction, 0)

    gt_bin = (ground_truth > 0.5).astype(np.uint8)
    pred_bin = (prediction > threshold).astype(np.uint8)
    
    if np.sum(gt_bin) == 0 and np.sum(pred_bin) == 0:
        return float("NaN") if nan_for_nonexisting else 0.0

    intersection = np.sum(gt_bin * pred_bin)
    return (2.0 * intersection) / (np.sum(gt_bin) + np.sum(pred_bin) + 1e-7)

def hausdorff_distance_95(
    ground_truth,
    prediction,
    mask=None,
    threshold=0.5,
    voxelspacing=None,  
    nan_for_nonexisting=True
):
    if mask is not None:
        ground_truth = np.where(mask, ground_truth, 0)
        prediction = np.where(mask, prediction, 0)

    gt_bin = (ground_truth > 0.5).astype(np.uint8)
    pred_bin = (prediction > threshold).astype(np.uint8)

    if np.sum(gt_bin) == 0 or np.sum(pred_bin) == 0:
        return float("NaN") if nan_for_nonexisting else 0.0
   
    return metric.hd95(gt_bin, pred_bin, voxelspacing=voxelspacing)

def centerline_dice(
    ground_truth,
    prediction,
    mask=None,
    threshold=0.5,
    nan_for_nonexisting=True
):
    if mask is not None:
        ground_truth = np.where(mask, ground_truth, 0)
        prediction = np.where(mask, prediction, 0)
    
    gt_bin = (ground_truth > 0.5).astype(np.uint8)
    pred_bin = (prediction > threshold).astype(np.uint8)
    
    if np.sum(gt_bin) == 0 or np.sum(pred_bin) == 0:
        return float("NaN") if nan_for_nonexisting else 0.0
    
    try:
        skeleton_gt = skeletonize(gt_bin)
        skeleton_pred = skeletonize(pred_bin)
    except ValueError:
        return float("NaN")

    intersection = np.sum(skeleton_gt * skeleton_pred)
    return (2.0 * intersection) / (np.sum(skeleton_gt) + np.sum(skeleton_pred) + 1e-7)


def compute_classification_metrics(
    ground_truth,
    prediction,
    mask=None,
    threshold=0.5,
):
    if mask is not None:
        ground_truth = ground_truth[mask]
        prediction = prediction[mask]
    ground_truth = np.where(ground_truth > 0.5, 1, 0)
    prediction = np.where(prediction > threshold, 1, 0)
    try:
        tn, fp, fn, tp = confusion_matrix(ground_truth, prediction).ravel()
    except ValueError:
        print('Value error excepted')
        tn, fp, fn, tp = 1, 0, 0, 1

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    f1 = 2 * tp / (2 * tp + fp + fn)

    return sensitivity, specificity, accuracy, f1


def get_vessel_tree_mask(ground_truth):
    unknown = ground_truth[:, :, 1] - ground_truth[:, :, 0] - ground_truth[:, :, 2]
    unknown[unknown < 0] = 0
    crossings = ground_truth[:, :, 0] * ground_truth[:, :, 1] * ground_truth[:, :, 2]
    unk = crossings + unknown

    mask = ground_truth[:, :, 1] - unk

    return mask, unknown, crossings


def get_topo_a_v_metrics(samples, model, gt_key, n_paths):
    infs_a = []
    corrs_a = []
    infs_v = []
    corrs_v = []
    for sample in samples.values():
        print('.', end='', flush=True)
        img = io.imread(sample[model]) / 255.0
        ground = np.round(io.imread(sample[gt_key]) / 255.0)

        inf_a, _, corr_a = topo_metric(ground[:, :, 0], img[:, :, 0], 0.5, n_paths)
        infs_a.append(inf_a/n_paths)
        corrs_a.append(corr_a/n_paths)

        inf_v, _, corr_v = topo_metric(ground[:, :, 2], img[:, :, 2], 0.5, n_paths)
        infs_v.append(inf_v/n_paths)
        corrs_v.append(corr_v/n_paths)
    return {
        'inf_a': infs_a,
        'corr_a': corrs_a,
        'inf_v': infs_v,
        'corr_v': corrs_v,
    }


def split_dict(d, n):
    step = len(d) // n
    splits = []
    keys = list(d.keys())
    for i in range(0, len(d), step):
        splits.append({k: d[k] for k in keys[i:i+step]})
    return splits


class TopoAVMetrics:
    def __init__(self, model, gt_key, n_paths):
        self.model = model
        self.gt_key = gt_key
        self.n_paths = n_paths

    def __call__(self, samples):
        return get_topo_a_v_metrics(samples, self.model, self.gt_key, self.n_paths)


def mav(
    dataset: Namespace,
    model: str,
    save_path: Union[str, Path],
    verbose: bool = True,
    predicted_only: bool = False,
    n_paths: int = 100, # 1000 for formal testing
    gt_key: str = 'gt',
    masks_key: str = 'masks',
) -> dict:
    h, w = dataset.shape
    samples = dataset.samples
    shape = (len(dataset.samples), h,w)
    all_preds_a = np.zeros(shape, np.float32)
    all_preds_v = np.zeros(shape, np.float32)
    all_gts_v = np.zeros(shape, np.float32)    
    all_gts = np.zeros(shape, np.float32)
    all_pred_vessels = np.zeros(shape, np.float32)
    all_gt_vessels = np.zeros(shape, np.float32)
    all_mask_vessels = np.zeros(shape, bool)
    
    all_preds = np.zeros(shape, np.float32)
    all_masks = np.zeros(shape, bool)
    
    i = 0
    infs_a = []
    corrs_a = []
    infs_v = []
    corrs_v = []
    for sample in tqdm(samples.values()): 
        img = io.imread(sample[model]) / 255.0
        if img.shape[0] != h or img.shape[1] != w:
            img = resize(img, (h, w), order=2, anti_aliasing=True, preserve_range=True)
        ground = io.imread(sample[gt_key]) / 255.0
        if ground.shape[0] != h or ground.shape[1] != w:
            ground = resize(ground, (h, w), order=0, anti_aliasing=True, preserve_range=True)
        ground = np.round(ground)
        
        ground_bool = ground.astype(bool)

        ch0 = ground_bool[:, :, 0]
        ch1 = ground_bool[:, :, 1]
        ch2 = ground_bool[:, :, 2]
        
        ground[:, :, 0] = np.max([ch0, ch1], axis=0)
        ground[:, :, 1] = np.max([ch0, ch1, ch2], axis=0)
        ground[:, :, 2] = np.max([ch1, ch2], axis=0)
        
        fov_mask = io.imread(sample[masks_key])
        if fov_mask.shape[0] != h or fov_mask.shape[1] != w:
            fov_mask = resize(fov_mask, (h, w), order=0, anti_aliasing=True, preserve_range=True)
        if len(fov_mask.shape) == 3:
            fov_mask = np.mean(fov_mask, axis=2)
        fov_mask = fov_mask > 0.5
        
        mask, _, _ = get_vessel_tree_mask(ground)
        mask = mask > 0.5
        
        if n_paths > 0:
            inf_a, _, corr_a = topo_metric(ground[:, :, 0], img[:, :, 0], 0.5, n_paths)
            infs_a.append(inf_a/n_paths)
            corrs_a.append(corr_a/n_paths)

            inf_v, _, corr_v = topo_metric(ground[:, :, 2], img[:, :, 2], 0.5, n_paths)
            infs_v.append(inf_v/n_paths)
            corrs_v.append(corr_v/n_paths)
        else:
            infs_a.append(.5)
            corrs_a.append(.5)
            infs_v.append(.5)
            corrs_v.append(.5)

        rand = np.random.random(img.shape[:2])  
        condition = (img[:, :, 0] == 0) & (img[:, :, 2] == 0) & mask
        img[condition, 0] = rand[condition]
        img[condition, 2] = 1 - rand[condition]
        pred_indices = 1 - np.argmax(img[:, :, [0,2]], axis=2)

        if predicted_only:
            mask = mask * img[:, :, 1] > 0.5

        all_preds_a[i] = img[:, :, 0]
        all_pred_vessels[i] = img[:, :, 1]
        all_preds_v[i] = img[:, :, 2]
        
        all_gts[i] = ground[:, :, 0]
        all_gt_vessels[i] = ground[:, :, 1]
        all_gts_v[i] = ground[:, :, 2]
        
        all_preds[i] = pred_indices 
        all_masks[i] = mask
        
        all_mask_vessels[i] = fov_mask
        
        i += 1
    print("------------Start Evaluation------------")
    
    dsc_bv = dice_coefficient(all_gt_vessels, all_pred_vessels, all_mask_vessels, threshold=0.5)
    hd95_bv = hausdorff_distance_95(all_gt_vessels, all_pred_vessels, all_mask_vessels, threshold=0.5)
    cldice_bv = centerline_dice(all_gt_vessels, all_pred_vessels, all_mask_vessels, threshold=0.5)

    dsc_a = dice_coefficient(all_gts, all_preds_a, all_mask_vessels, threshold=0.5)
    dsc_v = dice_coefficient(all_gts_v, all_preds_v, all_mask_vessels, threshold=0.5)
    
    all_preds_a = all_preds_a.flatten()
    all_preds_v = all_preds_v.flatten()
    all_gts_v = all_gts_v.flatten()
    all_gts = all_gts.flatten()
    all_preds = all_preds.flatten()
    all_masks = all_masks.flatten()
    all_gt_vessels = all_gt_vessels.flatten()
    all_pred_vessels = all_pred_vessels.flatten()
    all_mask_vessels = all_mask_vessels.flatten()

    sens_a, spec_a, acc_a, _ = compute_classification_metrics(all_gts, all_preds_a, all_mask_vessels)
    sens_v, spec_v, acc_v, _ = compute_classification_metrics(all_gts_v, all_preds_v, all_mask_vessels)

    inf_a = [np.mean(infs_a), np.std(infs_a)]
    corr_a = [np.mean(corrs_a), np.std(corrs_a)]
    inf_v = [np.mean(infs_v), np.std(infs_v)]
    corr_v = [np.mean(corrs_v), np.std(corrs_v)]
    

    if verbose:
        print('=' * 30)
        print(model)
        print('-' * 30)
        print("dsc_bv:", dsc_bv)
        print("hd95_bv:", hd95_bv)
        print("cldice_bv:", cldice_bv)
        print('-' * 30)
        print("dsc_a:", dsc_a)
        print('Sens A:', sens_a)
        print('Spec A:', spec_a)
        print('Acc A:', acc_a)
        print('Inf A:', inf_a)
        print('Corr A:', corr_a)
        print('-' * 30)
        print("dsc_v:", dsc_v)
        print('Sens V:', sens_v)
        print('Spec V:', spec_v)
        print('Acc V:', acc_v)
        print('Inf V:', inf_v)
        print('Corr V:', corr_v)
        print('-' * 30)

    partial_results = {
        'dsc_bv': dsc_bv,
        'hd95_bv': hd95_bv,
        'cldice_bv': cldice_bv,
        
        'dsc_a': dsc_a,
        'Sens A': sens_a,
        'Spec A': spec_a,
        'Acc A': acc_a,
        'Inf A': inf_a,
        'Corr A': corr_a,
        
        'dsc_v': dsc_v,
        'Sens V': sens_v,
        'Spec V': spec_v,
        'Acc V': acc_v,
        'Inf V': inf_v,
        'Corr V': corr_v,
    }
    return partial_results


test_factory = {
    'mav': mav
}