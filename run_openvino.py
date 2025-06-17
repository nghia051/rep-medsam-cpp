from openvino.runtime import Core
import numpy as np
import cv2
import torch
from PIL import Image

core = Core()

def load_encoder(model_path: str):
    """
    Load the encoder model from the specified path.
    """
    encoder_model = core.read_model(model_path)
    encoder_compiled = core.compile_model(encoder_model, "CPU")

    return encoder_compiled

def load_decoder(model_path: str):
    """
    Load the decoder model from the specified path.
    """
    decoder_model = core.read_model(model_path)
    decoder_compiled = core.compile_model(decoder_model, "CPU")

    return decoder_compiled

def load_models(encoder_path: str, decoder_path: str):
    encoder_compiled = load_encoder(encoder_path)
    decoder_compiled = load_decoder(decoder_path)

    print("Encoder and Decoder models loaded successfully.")
    print("=====================================")

    # Check encoder input/output names
    encoder_input = encoder_compiled.input(0)
    encoder_output = encoder_compiled.output(0)

    print("Encoder Input:", encoder_input.any_name, encoder_input.get_partial_shape())
    print("Encoder Output:", encoder_output.any_name, encoder_output.get_partial_shape())

    # Check decoder input/output names
    decoder_inputs = decoder_compiled.inputs
    decoder_output = decoder_compiled.output(0)
    for i, inp in enumerate(decoder_inputs):
        print(f"Decoder Input {i}:", inp.any_name, inp.shape)
    print("Decoder Output:", decoder_output.any_name, decoder_output.shape)

    print("=====================================")
    
    return encoder_compiled, decoder_compiled

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

encoder_path = "./output_model/encoder.xml"
decoder_path = "./output_model/decoder.xml"

encoder_compiled, decoder_compiled = load_models(encoder_path, decoder_path)

def nomormalize8(I: np.ndarray) -> np.ndarray:
    mn = I.min()
    mx = I.max()

    mx -= mn

    I = (I - mn) / mx
    I = I * 255
    
    return I.astype(np.uint8)

def load_and_prepare_image(img_npz_file: str, target_size=256):
    npz_data = np.load(img_npz_file, 'r', allow_pickle=True)
    img_3c = npz_data['imgs'] # (H, W, 3)
    assert np.max(img_3c)<256, f'input data should be in range [0, 255], but got {np.unique(img_3c)}'
    H, W = img_3c.shape[:2]
    boxes = npz_data['boxes']

    print(f"Image shape: {img_3c.shape}, dtype: {img_3c.dtype}, min: {img_3c.min()}, max: {img_3c.max()}")  # should be [H, W, 3]
    print(f"Boxes shape: {boxes.shape}, dtype: {boxes.dtype}, min: {boxes.min()}, max: {boxes.max()}")  # should be [1, 4]
    ## preprocessing
    img_256 = resize_longest_side(img_3c, target_size)
    print(f"Image shape after resizing: {img_256.shape}, dtype: {img_256.dtype}, min: {img_256.min()}, max: {img_256.max()}")  # should be [256, 256, 3]

    img_256_padded = pad_image(img_256, target_size)
    print(f"Image shape after padding: {img_256_padded.shape}, dtype: {img_256_padded.dtype}, min: {img_256_padded.min()}, max: {img_256_padded.max()}")

    img_256_padded = img_256_padded.astype(np.float32) / 255.0  # Normalize to [0, 1]
    print(f"Image shape after normalization: {img_256_padded.shape}, dtype: {img_256_padded.dtype}, min: {img_256_padded.min()}, max: {img_256_padded.max()}")

    encoder_output = encoder_compiled([img_256_padded])
    encoder_result = encoder_output[encoder_compiled.output(0)]

    print("Encoder result shape:", encoder_result.shape)  # should be [1, 256, 64, 64]

    decoder_inputs = {
        decoder_compiled.input(0).any_name: encoder_result,
        decoder_compiled.input(1).any_name: boxes[0]
    }
    decoder_result = decoder_compiled(decoder_inputs)["masks"]

    print("Decoder result shape:", decoder_result.shape)  # should be [1, 256, 256]
    
    mask = decoder_result.squeeze()  # shape [256, 256]
    print(f"Mask shape: {mask.shape}, dtype: {mask.dtype}, min: {mask.min()}, max: {mask.max()}")

    mask_norm = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8) * 255
    print(f"Mask shape norm: {mask_norm.shape}, dtype: {mask_norm.dtype}, min: {mask_norm.min()}, max: {mask_norm.max()}")

    mask_uint8 = mask_norm.astype(np.uint8)
    print(f"Mask shape uint8: {mask_uint8.shape}, dtype: {mask_uint8.dtype}, min: {mask_uint8.min()}, max: {mask_uint8.max()}")

    # img = Image.fromarray(mask_uint8)
    # img.show()

    cv2.imwrite("output_mask.png", mask_uint8)

    # Normalize mask to [0, 1] range
    # mask = nomormalize8(mask)
    # print(f"Mask shape after normalization: {mask.shape}, dtype: {mask.dtype}, min: {mask.min()}, max: {mask.max()}")

if __name__ == '__main__':
    load_and_prepare_image("./test_demo/imgs/2DBox_CXR_demo.npz", target_size=256)

    
