import torch

import torchvision
from torch import nn


DINO_MEAN = [0.485, 0.456, 0.406]
DINO_STD  = [0.229, 0.224, 0.225]

class DinoTxtWrapper(nn.Module):
    """
    Minimal wrapper to keep the rest of your pipeline intact:
    - exposes encode_image()
    - applies resize(224) + ImageNet normalize
    """
    def __init__(self):
        super().__init__()
        self.model = torch.hub.load(
            "facebookresearch/dinov2",
            "dinov2_vitl14_reg4_dinotxt_tet1280d20h24l",
        ).eval().to("cuda")

        self.process = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.Normalize(mean=DINO_MEAN, std=DINO_STD),
            ]
        )

        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,3,H,W) float in [0,1]
        x = self.process(x)
        if hasattr(self.model, "encode_image"):
            feat = self.model.encode_image(x)
        else:
            out = self.model(x)
            feat = out[0] if isinstance(out, (tuple, list)) else out
        return feat  # [N, D]
    
    
    @staticmethod
    def get_feature_map(seg_map, feature_map):
        H, W = seg_map.shape

        y, x = torch.meshgrid(torch.arange(0, H, device='cuda'), torch.arange(0, W, device='cuda'))
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)

        seg = seg_map[y, x].squeeze(-1).long()
        mask = seg != -1
        _point_feature = feature_map[seg].squeeze(0)
        mask = mask.reshape(H, W)
        
        point_feature = _point_feature.reshape(H, W, -1).permute(2, 0, 1)
       
        return point_feature, mask


