import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from transformers import pipeline
from PIL import Image
import torchvision
from torchvision.transforms import v2


def make_transform(resize_size: int = 256):
    to_tensor = v2.ToImage()
    resize = v2.Resize((resize_size, resize_size), antialias=True)
    to_float = v2.ToDtype(torch.float32, scale=True)
    normalize = v2.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    return v2.Compose([to_tensor, resize, to_float, normalize])


class DinoExtractor(nn.Module):
    def __init__(self):
        super().__init__()
                
        self.feature_extractor = pipeline(
            model="facebook/dinov3-vitb16-pretrain-lvd1689m",
            task="image-feature-extraction", 
            device='cuda:0',
        )

    @torch.no_grad()
    def encode_image(self, x):
        x = x.permute(0, 2, 3, 1).cpu().numpy()  # [B, H, W, 3]
        B, H, W, _ = x.shape

        if x.dtype != "uint8":
            x = (x * 255).clip(0, 255).astype("uint8")

        images = [Image.fromarray(img, mode="RGB") for img in x]

        features = self.feature_extractor(images)
        features_tensor = torch.tensor(features, dtype=torch.float32)

        B, _, N, C = features_tensor.shape
        h = H // 16
        w = W // 16
        features_tensor = features_tensor[:, :, 5:, :]
        features_tensor = features_tensor.reshape(B, h, w, C)    # (14, 14, 768)

        return features_tensor # [N, h, w, D]
    
    @staticmethod
    def get_feature_map(seg_map, patch_feats):
        """
        seg_map: (H, W), segment id for each pixel, -1 indicates ignore
        patch_feats: (1, N_patches, D), patch-level feature map from DINO
        returns:
            dense_feature: (D, H, W)
            mask: (H, W), True for valid pixels
        """

        H, W = seg_map.shape
        seg_ids = seg_map.unique()
        seg_ids = seg_ids[seg_ids != -1]  # Ignore -1 values

        D = patch_feats.shape[-1]  # Feature dimension, e.g., 1280

        # Initialize dense feature map
        dense_feature = torch.zeros(D, H, W, device=patch_feats.device)
        mask = seg_map != -1

        for seg_id in seg_ids:
            # Current segment mask
            seg_mask = seg_map == seg_id  # (H, W), bool

            if seg_mask.sum() == 0:
                continue

            # Bounding box of the segment
            coords = seg_mask.nonzero(as_tuple=False)  # (N_pixels, 2)
            y1, x1 = coords.min(0)[0]
            y2, x2 = coords.max(0)[0] + 1

            # Patch-level feature map: (D, H_patch, W_patch)
            patch_map = patch_feats[seg_id].permute(2, 0, 1)

            # Upsample to bounding box size
            seg_feats = F.interpolate(patch_map.unsqueeze(0), size=(y2-y1, x2-x1),
                                    mode='bilinear', align_corners=False).squeeze(0)  # (D, h_box, w_box)

            # Only write back to pixels belonging to the current segment
            seg_mask_crop = seg_mask[y1:y2, x1:x2]  # (h_box, w_box)
            dense_feature[:, y1:y2, x1:x2][:, seg_mask_crop] = seg_feats[:, seg_mask_crop]

        return dense_feature, mask
