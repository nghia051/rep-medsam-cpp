from openvino.runtime import Core
from PIL import Image
from os.path import join, basename
from glob import glob
from tqdm import tqdm
from time import time
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
import cv2


#%% set seeds
torch.set_float32_matmul_precision('high')
torch.manual_seed(2024)
torch.cuda.manual_seed(2024)
np.random.seed(2024)

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
    # decoder_inputs = decoder_compiled.inputs
    # decoder_output = decoder_compiled.output(0)
    # for i, inp in enumerate(decoder_inputs):
    #     print(f"Decoder Input {i}:", inp.any_name, inp.shape)
    # print("Decoder Output:", decoder_output.any_name, decoder_output.shape)

    print("=====================================")

    return encoder_compiled, decoder_compiled

def show_mask(mask, ax, mask_color=None, alpha=0.5):
    """
    show mask on the image

    Parameters
    ----------
    mask : numpy.ndarray
        mask of the image
    ax : matplotlib.axes.Axes
        axes to plot the mask
    mask_color : numpy.ndarray
        color of the mask
    alpha : float
        transparency of the mask
    """
    if mask_color is not None:
        color = np.concatenate([mask_color, np.array([alpha])], axis=0)
    else:
        color = np.array([251/255, 252/255, 30/255, alpha])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, edgecolor='blue'):
    """
    show bounding box on the image

    Parameters
    ----------
    box : numpy.ndarray
        bounding box coordinates in the original image
    ax : matplotlib.axes.Axes
        axes to plot the bounding box
    edgecolor : str
        color of the bounding box
    """
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor=edgecolor, facecolor=(0,0,0,0), lw=2))  

def save_overlay_image(img, segs, boxes, output_path):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(img)
    ax[1].imshow(img)
    ax[0].set_title("Image")
    ax[1].set_title("LiteMedSAM Segmentation")
    ax[0].axis('off')
    ax[1].axis('off')

    for i, box in enumerate(boxes):
        color = np.random.rand(3)  # Random color for each box
        box_viz = box
        show_box(box_viz, ax[1], edgecolor='blue')
        show_mask((segs == i+1).astype(np.uint8), ax[1], mask_color=color)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

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

def resize_box_to_256(box, original_size):
    """
    the input bounding box is obtained from the original image
    here, we rescale it to the coordinates of the resized image

    Parameters
    ----------
    box : numpy.ndarray
        bounding box coordinates in the original image
    original_size : tuple
        the original size of the image

    Returns
    -------
    numpy.ndarray
        bounding box coordinates in the resized image
    """
    new_box = np.zeros_like(box)
    ratio = 256 / max(original_size)
    for i in range(len(box)):
        new_box[i] = int(box[i] * ratio)

    return new_box

def postprocess_masks(masks, new_size, original_size):
    # Crop
    masks = masks[..., :new_size[0], :new_size[1]]
    # Resize
    masks = F.interpolate(
        masks,
        size=(original_size[0], original_size[1]),
        mode="bilinear",
        align_corners=False,
    )

    return masks

model_name = "rep_medsam_fixed"
encoder_path = "./openvino_models/" + model_name + "/encoder.xml"
decoder_path = "./openvino_models/" + model_name + "/decoder.xml"

encoder_compiled, decoder_compiled = load_models(encoder_path, decoder_path)

def compute_image_embeddings(img_256_tensor):
    image_embeddings_numpy = encoder_compiled([img_256_tensor])["image_embeddings"]

    image_embeddings_tensor = torch.from_numpy(image_embeddings_numpy)

    return image_embeddings_tensor

def compute_segmentation_masks(image_embeddings_tensor, boxes, new_size, original_size):
    segs = np.zeros(original_size, dtype=np.uint16)
    
    for idx in range(len(boxes)):
        box256 = resize_box_to_256(boxes[idx], original_size)
        box256 = box256[None, ...] # (1, 4)

        box_tensor = torch.tensor(box256, dtype=torch.float32, device="cpu")  # shape [1, 4]

        low_res_logits = decoder_compiled([image_embeddings_tensor, box_tensor])["masks"]
        low_res_logits = torch.from_numpy(low_res_logits)

        low_res_pred = postprocess_masks(low_res_logits, new_size, original_size)
        low_res_pred = torch.sigmoid(low_res_pred)
        low_res_pred = low_res_pred.squeeze().cpu().numpy()
        medsam_seg = (low_res_pred > 0.5).astype(np.uint8)

        segs[medsam_seg > 0] = idx + 1        

    return segs

def infer_2D(img_npz_file: str):
    npz_data = np.load(img_npz_file, 'r', allow_pickle=True)
    img_3c = npz_data['imgs'] # (H, W, 3)
    assert np.max(img_3c)<256, f'input data should be in range [0, 255], but got {np.unique(img_3c)}'

    H, W = img_3c.shape[:2]
    boxes = npz_data['boxes'].to_device("cpu")
    
    # preprocess image
    img_256 = resize_longest_side(img_3c, target_length=256)
    newh, neww = img_256.shape[:2]

    img_256_norm = (img_256 - img_256.min()) / np.clip(
        img_256.max() - img_256.min(), a_min=1e-8, a_max=None
    )
    # print(f"Image shape after normalization: {img_256_norm.shape}, dtype: {img_256_norm.dtype}, min: {img_256_norm.min()}, max: {img_256_norm.max()}")  # should be [256, 256, 3]

    img_256_padded = pad_image(img_256_norm, 256)
    # print(f"Image shape after padding: {img_256_padded.shape}, dtype: {img_256_padded.dtype}, min: {img_256_padded.min()}, max: {img_256_padded.max()}")

    img_256_tensor = torch.tensor(img_256_padded).float().permute(2, 0, 1).unsqueeze(0).to("cpu")  # shape [1, 3, 256, 256]
    # print(f"Image tensor shape: {img_256_tensor.shape}, dtype: {img_256_tensor.dtype}, min: {img_256_tensor.min()}, max: {img_256_tensor.max()}")

    image_embeddings_tensor = compute_image_embeddings(img_256_tensor)  # shape [1, 256, 64, 64]

    segs = compute_segmentation_masks(image_embeddings_tensor, boxes,
                                      new_size=(newh, neww), original_size=(H, W))

    np.savez_compressed(
        join("./output/" + model_name + "/segs/", basename(img_npz_file)),
        segs=segs,
    )

    # save overlay
    # save_overlay_image(img_3c, segs, boxes, join("./output/" + model_name + "/overlay/", basename(img_npz_file).replace('.npz', '.png')))

if __name__ == '__main__':
    img_npz_files = sorted(glob(join("./dataset/imgs/", '*.npz'), recursive=True))

    print(f"Found {len(img_npz_files)} image files to process.")

    for img_npz_file in tqdm(img_npz_files):
        if basename(img_npz_file).startswith('2D'):
            start_time = time()
            infer_2D(img_npz_file)
            end_time = time()
            print('file name:', basename(img_npz_file), 'time cost:', np.round(end_time - start_time, 4))
