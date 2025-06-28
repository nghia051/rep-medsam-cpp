import numpy as np
from pathlib import Path
from tqdm import tqdm
from os.path import join, basename
from evaluation.SurfaceDice import compute_surface_distances, compute_surface_dice_at_tolerance

def load_pred_mask(npz_file: str):
    data = np.load(npz_file, "r")
    if "segs" not in data.keys():
        raise ValueError("No data found in the npz file.")
    
    return data["segs"]

def load_gt_mask(npz_file: str):
    data = np.load(npz_file, "r")
    if "gts" not in data.keys() and "segs" not in data.keys():
        raise ValueError("No data found in the npz file.")
    
    return data["gts"] if "gts" in data.keys() else data["segs"]

def compute_dice_coefficient(mask_gt: np.ndarray, mask_pred: np.ndarray):
    if mask_gt.shape != mask_pred.shape:
        raise ValueError("Shape mismatch: mask_gt shape {} vs mask_pred shape {}".format(mask_gt.shape, mask_pred.shape))
    
    volume_sum = mask_gt.sum() + mask_pred.sum()
    if volume_sum == 0:
        return np.NaN
    volume_intersect = (mask_gt & mask_pred).sum()
    
    return 2*volume_intersect / volume_sum

def compute_multi_class_dsc(gt, seg):
    dsc = []
    for i in range(1, gt.max()+1):
        gt_i = gt == i
        seg_i = seg == i

        dsc.append(np.float64(compute_dice_coefficient(gt_i, seg_i)))

    return np.mean(dsc)

def compute_multi_class_nsd(gt, seg, spacing, tolerance=2.0):
    nsd = []
    for i in range(1, gt.max()+1):
        gt_i = gt == i
        seg_i = seg == i
        surface_distance = compute_surface_distances(
            gt_i, seg_i, spacing_mm=spacing
        )
        nsd.append(np.float64(compute_surface_dice_at_tolerance(surface_distance, tolerance)))

    return np.mean(nsd)

if __name__ == '__main__':
    lite_medsam_seg_path = Path("./output/lite_medsam/segs")
    lite_medsam_cpp_seg_path = Path("./output/lite_medsam_cpp/segs")

    test_file = "2DBox_Microscope_0009.npz"

    gt_path = Path("./dataset/gts")

    lite_medsam_seg_mask = load_pred_mask(lite_medsam_seg_path / test_file)
    lite_medsam_cpp_seg_mask = load_pred_mask(lite_medsam_cpp_seg_path / test_file)
    gt_mask = load_gt_mask(gt_path / test_file)

    # number of unique labels in the masks
    print("num unique lite_medsam_seg_mask:", len(np.unique(lite_medsam_seg_mask)))
    print("num unique lite_medsam_cpp_seg_mask:", len(np.unique(lite_medsam_cpp_seg_mask)))
    print("num unique gt_mask:", len(np.unique(gt_mask)))

    