import numpy as np
from PIL import Image

def load_npz_mask(npz_file: str):
    data = np.load(npz_file)
    data_key = list(data.keys())[0] if len(data.keys()) > 0 else None
    if data_key is None:
        raise ValueError("No data found in the npz file.")
    mask = data[data_key]
    return mask

def dice_coefficient(pred: np.ndarray, gt: np.ndarray):
    if pred.shape != gt.shape:
        raise ValueError("Shape mismatch: pred shape {} vs gt shape {}".format(pred.shape, gt.shape))
    
    if pred.dtype != gt.dtype:
        raise ValueError("Data type mismatch: pred dtype {} vs gt dtype {}".format(pred.dtype, gt.dtype))

    intersection = np.sum(pred * gt)
    dice = (2. * intersection) / (np.sum(pred) + np.sum(gt) + 1e-8)  # small epsilon to avoid division by zero

    return dice

def nomormalize8(I: np.ndarray) -> np.ndarray:
    mn = I.min()
    mx = I.max()

    mx -= mn

    I = (I - mn) / mx
    I = I * 255
    
    return I.astype(np.uint8)

def visualize_npz(npz_file: str):
    data = np.load(npz_file)    
    print(f"Keys in the npz file: {list(data.keys())}")

    data_key = list(data.keys())[0] if len(data.keys()) > 0 else None
    if data_key is None:
        print("No data found in the npz file.")
        return

    array = data[data_key]
    print(f"Shape of the array before normalizing: {array.shape}, dtype: {array.dtype}")

    # array = nomormalize8(array)
    # print(f"Shape of the array after normalizing: {array.shape}, dtype: {array.dtype}")

    img = Image.fromarray(array)
    img.show()

if __name__ == '__main__':
    pred_file = './test_demo/segs/2DBox_CXR_demo.npz'

    visualize_npz(pred_file)