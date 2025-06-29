from typing import List, Optional

import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.transforms.v2 import functional as F2


class EncoderOnnxModel(nn.Module):
    def __init__(
        self,
        image_encoder: nn.Module,
        preprocess_image: bool = True,
        image_encoder_input_size: int = 256,
        scale_image: bool = True,
        normalize_image: bool = False,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
        interpolation: str = "bilinear",
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.preprocess_image = preprocess_image
        self.image_encoder_input_size = image_encoder_input_size
        self.scale_image = scale_image
        self.normalize_image = normalize_image
        self.pixel_mean = pixel_mean
        self.pixel_std = pixel_std
        self.interpolation = interpolation

    @torch.no_grad()
    def forward(
        self,
        image: torch.Tensor,
    ) -> torch.Tensor:
        """
        image: (H, W, 3)
        """
        image = image.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)

        return self.image_encoder(image)

    @staticmethod
    def get_preprocess_shape(
        original_size: torch.Tensor,
        long_side_length: int,
    ) -> torch.Tensor:
        original_size = original_size.to(torch.float32)
        scale = long_side_length / torch.max(original_size)
        new_size = scale * original_size
        new_size = torch.floor(new_size + 0.5).to(torch.int16)
        return new_size
