import torch
import torch.nn as nn
import torch.nn.functional as F

from segment_anything.modeling import MaskDecoder, PromptEncoder


class DecoderOnnxModel(nn.Module):
    def __init__(
        self,
        mask_decoder: MaskDecoder,
        prompt_encoder: PromptEncoder,
    ):
        super().__init__()
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder

    @torch.no_grad()
    def forward(
        self,
        image_embeddings: torch.Tensor,
        boxes: torch.Tensor,
    ):
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=None,
            boxes=boxes,
            masks=None,
        )

        low_res_logits, _ = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        return low_res_logits
