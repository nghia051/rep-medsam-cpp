import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import openvino as ov
from pathlib import Path
import io
import onnx
from onnxsim import simplify
from onnxruntime.quantization import QuantType
from onnxruntime.quantization.quantize import quantize_dynamic
import onnxruntime as ort

from segment_anything.modeling import MaskDecoder, PromptEncoder, TwoWayTransformer
from tiny_vit_sam import TinyViT
import argparse

from models.onnx import DecoderOnnxModel


#%% set seeds
torch.set_float32_matmul_precision('high')
torch.manual_seed(2024)
torch.cuda.manual_seed(2024)
np.random.seed(2024)

parser = argparse.ArgumentParser()

parser.add_argument(
    '-lite_medsam_checkpoint_path',
    type=str,
    default="work_dir/LiteMedSAM/lite_medsam.pth",
    help='path to the checkpoint of MedSAM-Lite',
)
parser.add_argument(
    '-device',
    type=str,
    default="cpu",
    help='device to run the inference',
)

args = parser.parse_args()

lite_medsam_checkpoint_path = args.lite_medsam_checkpoint_path
device = torch.device(args.device)

class MedSAM_Lite(nn.Module):
    def __init__(
            self, 
            image_encoder, 
            mask_decoder,
            prompt_encoder
        ):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder

    def forward(self, image, box_np):
        image_embedding = self.image_encoder(image) # (B, 256, 64, 64)
        # do not compute gradients for prompt encoder
        with torch.no_grad():
            box_torch = torch.as_tensor(box_np, dtype=torch.float32, device=image.device)
            if len(box_torch.shape) == 2:
                box_torch = box_torch[:, None, :] # (B, 1, 4)

        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=None,
            boxes=box_np,
            masks=None,
        )
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=image_embedding, # (B, 256, 64, 64)
            image_pe=self.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
            multimask_output=False,
          ) # (B, 1, 256, 256)

        return low_res_masks

    @torch.no_grad()
    def postprocess_masks(self, masks, new_size, original_size):
        """
        Do cropping and resizing

        Parameters
        ----------
        masks : torch.Tensor
            masks predicted by the model
        new_size : tuple
            the shape of the image after resizing to the longest side of 256
        original_size : tuple
            the original shape of the image

        Returns
        -------
        torch.Tensor
            the upsampled mask to the original size
        """
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

medsam_lite_image_encoder = TinyViT(
    img_size=256,
    in_chans=3,
    embed_dims=[
        64, ## (64, 256, 256)
        128, ## (128, 128, 128)
        160, ## (160, 64, 64)
        320 ## (320, 64, 64) 
    ],
    depths=[2, 2, 6, 2],
    num_heads=[2, 4, 5, 10],
    window_sizes=[7, 7, 14, 7],
    mlp_ratio=4.,
    drop_rate=0.,
    drop_path_rate=0.0,
    use_checkpoint=False,
    mbconv_expand_ratio=4.0,
    local_conv_size=3,
    layer_lr_decay=0.8
)

medsam_lite_prompt_encoder = PromptEncoder(
    embed_dim=256,
    image_embedding_size=(64, 64),
    input_image_size=(256, 256),
    mask_in_chans=16
)

medsam_lite_mask_decoder = MaskDecoder(
    num_multimask_outputs=3,
        transformer=TwoWayTransformer(
            depth=2,
            embedding_dim=256,
            mlp_dim=2048,
            num_heads=8,
        ),
        transformer_dim=256,
        iou_head_depth=3,
        iou_head_hidden_dim=256,
)

medsam_lite_model = MedSAM_Lite(
    image_encoder = medsam_lite_image_encoder,
    mask_decoder = medsam_lite_mask_decoder,
    prompt_encoder = medsam_lite_prompt_encoder
)

lite_medsam_checkpoint = torch.load(lite_medsam_checkpoint_path, map_location='cpu')
medsam_lite_model.load_state_dict(lite_medsam_checkpoint)
medsam_lite_model.to(device)
medsam_lite_model.eval()

def convert_to_openvino(onnx_model: Path):
    model = ov.convert_model(onnx_model)
    ov.save_model(model, onnx_model.with_suffix(".xml"), compress_to_fp16=False)

def export_encoder(sam_model: MedSAM_Lite, export_optimized: bool = False, export_quantized: bool = False):
    dummy_inputs = torch.randn((1, 3, 256, 256,), dtype = torch.float32)

    output_file = Path("./new_model") / "encoder.onnx"

    buffer = io.BytesIO()
    torch.onnx.export(
        model=sam_model.image_encoder,
        args=dummy_inputs,
        f=buffer,
        export_params=True,
        verbose=False,
        opset_version=17,
        do_constant_folding=True, 
        input_names=['image'],
        output_names=['image_embeddings'],
    )
    buffer.seek(0, 0)

    # simplify the ONNX model
    onnx_model = onnx.load_model(buffer)
    onnx_model, success = simplify(onnx_model)
    assert success
    new_buffer = io.BytesIO()
    onnx.save(onnx_model, new_buffer)
    buffer = new_buffer
    buffer.seek(0, 0)

    with open(output_file, "wb") as f:
        f.write(buffer.read())
    
    convert_to_openvino(output_file)

    if export_optimized:
        optimized_output_file = output_file.with_suffix(".optimized.onnx")
        opt = ort.SessionOptions()
        opt.optimized_model_filepath = optimized_output_file.as_posix()
        opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        _ = ort.InferenceSession(output_file, opt, providers=["CPUExecutionProvider"])

    if export_quantized:
        quantized_output_file = output_file.with_suffix(".quantized.onnx")
        quantize_dynamic(
            model_input=output_file,
            model_output=quantized_output_file,
            per_channel=False,
            reduce_range=False,
            weight_type=QuantType.QUInt8,
        )

        convert_to_openvino(quantized_output_file)

def export_decoder(sam_model: MedSAM_Lite, export_optimized: bool = False, export_quantized: bool = False):
    onnx_model = DecoderOnnxModel(
        mask_decoder=sam_model.mask_decoder,
        prompt_encoder=sam_model.prompt_encoder,
    )

    embed_dim = onnx_model.prompt_encoder.embed_dim
    embed_size = onnx_model.prompt_encoder.image_embedding_size

    dummy_inputs = {
        "image_embeddings": torch.randn(1, embed_dim, *embed_size, dtype=torch.float32),
        "boxes": torch.rand((1, 4), dtype=torch.float32),
    }
    output_names = ["masks"]

    output_file = Path("./new_model") / "decoder.onnx"

    buffer = io.BytesIO()
    torch.onnx.export(
        model=onnx_model,
        args=dummy_inputs,
        f=buffer,
        export_params=True,
        verbose=False,
        opset_version=17,
        do_constant_folding=True, 
        input_names=list(dummy_inputs.keys()),
        output_names=output_names,
        dynamic_axes={
            "boxes": {0: "num_boxes"},
        },
    )
    buffer.seek(0, 0)

    # simplify the ONNX model
    onnx_model = onnx.load_model(buffer)
    onnx_model, success = simplify(onnx_model)
    assert success
    new_buffer = io.BytesIO()
    onnx.save(onnx_model, new_buffer)
    buffer = new_buffer
    buffer.seek(0, 0)

    with open(output_file, "wb") as f:
        f.write(buffer.read())
    
    convert_to_openvino(output_file)

    if export_optimized:
        optimized_output_file = output_file.with_suffix(".optimized.onnx")
        opt = ort.SessionOptions()
        opt.optimized_model_filepath = optimized_output_file.as_posix()
        opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        _ = ort.InferenceSession(output_file, opt, providers=["CPUExecutionProvider"])

    if export_quantized:
        quantized_output_file = output_file.with_suffix(".quantized.onnx")
        quantize_dynamic(
            model_input=output_file,
            model_output=quantized_output_file,
            per_channel=False,
            reduce_range=False,
            weight_type=QuantType.QUInt8,
        )

        convert_to_openvino(quantized_output_file)

if __name__ == '__main__':
    # print("Exporting encoder...")
    # export_encoder(medsam_lite_model,
    #                export_optimized=False,
    #                export_quantized=True)

    print("Exporting decoder...")
    export_decoder(medsam_lite_model,
                   export_optimized=False,
                   export_quantized=True)
