from openvino.runtime import Core

import numpy as np
import torch
import cv2

core = Core()

def load_encoder(model_path: str):
    encoder_model = core.read_model(model_path)
    encoder_compiled = core.compile_model(encoder_model, "CPU")

    return encoder_compiled

def resize_longest_side(image, target_length=256):
    """
    Resize image to target_length while keeping the aspect ratio
    Expects a numpy array with shape HxWxC in uint8 format.
    """
    oldh, oldw = image.shape[0], image.shape[1]
    scale = target_length * 1.0 / max(oldh, oldw)
    newh, neww = oldh * scale, oldw * scale
    neww, newh = int(neww + 0.5), int(newh + 0.5)
    target_size = (neww, newh)

    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

def pad_image(image, target_size=256):
    """
    Pad image to target_size
    Expects a numpy array with shape HxWxC in uint8 format.
    """
    # Pad
    h, w = image.shape[0], image.shape[1]
    padh = target_size - h
    padw = target_size - w
    if len(image.shape) == 3: ## Pad image
        image_padded = np.pad(image, ((0, padh), (0, padw), (0, 0)))
    else: ## Pad gt mask
        image_padded = np.pad(image, ((0, padh), (0, padw)))

    return image_padded

encoder_path = "./exported_model/encoder.xml"

encoder_compiled = load_encoder(encoder_path)

encoder_input = encoder_compiled.input(0)
encoder_output = encoder_compiled.output(0)

print("Encoder Input:", encoder_input.any_name, encoder_input.get_partial_shape())
print("Encoder Output:", encoder_output.any_name, encoder_output.get_partial_shape())

def run_encoder(img_npz_file: str, target_size=256):
    npz_data = np.load(img_npz_file, 'r', allow_pickle=True)
    img_3c = npz_data['imgs'] # (H, W, 3)
    assert np.max(img_3c)<256, f'input data should be in range [0, 255], but got {np.unique(img_3c)}'
    boxes = npz_data['boxes']
    

    print(f"Image shape: {img_3c.shape}, dtype: {img_3c.dtype}, min: {img_3c.min()}, max: {img_3c.max()}")  # should be [H, W, 3]
    print(f"Boxes shape: {boxes.shape}, dtype: {boxes.dtype}, min: {boxes.min()}, max: {boxes.max()}")  # should be [1, 4]
    
    ## preprocessing
    img_256 = resize_longest_side(img_3c, target_size)
    print(f"Image shape after resizing: {img_256.shape}, dtype: {img_256.dtype}, min: {img_256.min()}, max: {img_256.max()}")  # should be [256, 256, 3]

    img_256_norm = (img_256 - img_256.min()) / np.clip(
        img_256.max() - img_256.min(), a_min=1e-8, a_max=None
    )
    print(f"Image shape after normalization: {img_256_norm.shape}, dtype: {img_256_norm.dtype}, min: {img_256_norm.min()}, max: {img_256_norm.max()}")  # should be [256, 256, 3]

    img_256_padded = pad_image(img_256_norm, target_size)
    print(f"Image shape after padding: {img_256_padded.shape}, dtype: {img_256_padded.dtype}, min: {img_256_padded.min()}, max: {img_256_padded.max()}")

    img_256_tensor = torch.tensor(img_256_padded).float().permute(2, 0, 1).unsqueeze(0).to("cpu")  # shape [1, 3, 256, 256]
    print(f"Image tensor shape: {img_256_tensor.shape}, dtype: {img_256_tensor.dtype}, min: {img_256_tensor.min()}, max: {img_256_tensor.max()}")

    encoder_output = encoder_compiled([img_256_tensor])
    encoder_result = encoder_output[encoder_compiled.output(0)]
    print(f"Encoder output shape: {encoder_result.shape}, dtype: {encoder_result.dtype}")
    print(f"Encoder output type: {type(encoder_result)}")

if __name__ == '__main__':
    run_encoder("./test_demo/imgs/2DBox_CXR_demo.npz", target_size=256)
