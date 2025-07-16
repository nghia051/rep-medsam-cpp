import torch
import torch.nn as nn

class EncoderOnnxModel(nn.Module):
    def __init__(
        self,
        image_encoder: nn.Module,
    ):
        super().__init__()
        self.image_encoder = image_encoder

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
