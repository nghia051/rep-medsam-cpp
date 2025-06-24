import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import OrderedDict
import pandas as pd
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
    gt_path = "./dataset/gts/"
    seg_path = "./output/rep_medsam/segs/"
    output_csv = "./output/rep_medsam/seg_metrics.csv"

    seg_files = sorted(Path(seg_path).glob("*.npz"))

    seg_metrics = OrderedDict()
    seg_metrics["case"] = [] 
    seg_metrics["dsc"] = []
    seg_metrics["nsd"] = []

    for seg_file in tqdm(seg_files):
        gt_file = Path(gt_path) / seg_file.name
        if not gt_file.exists():
            print(f"GT file not found for {seg_file}, skipping.")
            continue
        
        segs_mask = load_pred_mask(seg_file)
        gts_mask = load_gt_mask(gt_file)

        dsc = compute_multi_class_dsc(gts_mask, segs_mask)

        # only compute nsd when dice > 0.2 because NSD is also low when dice is too low
        if dsc > 0.2:
            if seg_file.name.startswith("3D"):
                npz_gt = np.load(gt_file, "r")
                spacing = npz_gt["spacing"]
                nsd = compute_multi_class_nsd(gts_mask, segs_mask, spacing)
            else:
                spacing = [1.0, 1.0, 1.0]
                nsd = compute_multi_class_nsd(
                    np.expand_dims(gts_mask, -1), np.expand_dims(segs_mask, -1), spacing
                )
        else:
            nsd = 0.0

        seg_metrics["case"].append(seg_file.name)
        seg_metrics["dsc"].append(np.round(dsc, 4))
        seg_metrics["nsd"].append(np.round(nsd, 4))

    dsc_np = np.array(seg_metrics["dsc"])
    nsd_np = np.array(seg_metrics["nsd"])
    avg_dsc = np.mean(dsc_np[~np.isnan(dsc_np)])
    avg_nsd = np.mean(nsd_np[~np.isnan(nsd_np)])
    seg_metrics["case"].append("average")
    seg_metrics["dsc"].append(avg_dsc)
    seg_metrics["nsd"].append(avg_nsd)

    df = pd.DataFrame(seg_metrics)
    df.to_csv(output_csv, index=False, na_rep="NaN")

    print("Average DSC: {:.4f}".format(avg_dsc))
    print("Average NSD: {:.4f}".format(avg_nsd))
    