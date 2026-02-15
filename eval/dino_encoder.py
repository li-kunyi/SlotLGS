import torch
from torch import nn
import torch.nn.functional as F
from transformers import pipeline
from PIL import Image
import torchvision
from torchvision.transforms import v2
from transformers import AutoImageProcessor, AutoModel
import torch.nn.functional as F
import clip
from talk2dino.src.model import ProjectionLayer
import torch, os



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
    def __init__(self, device: str = "cuda"):
        super().__init__()

        # DINOv3
        self.processor = AutoImageProcessor.from_pretrained(
            "facebook/dinov3-vitb16-pretrain-lvd1689m"
        )
        self.model = AutoModel.from_pretrained(
            "facebook/dinov3-vitb16-pretrain-lvd1689m"
        ).cuda().eval()

        # Talk2DINO
        proj_name = 'dinov3_vitb_mlp_infonce'
        config_path = os.path.join("configs", f"{proj_name}.yaml")
        weights_path = os.path.join("weights", f"{proj_name}.pth")

        self.talk2dino = ProjectionLayer.from_config(config_path)
        self.talk2dino.load_state_dict(torch.load(weights_path, map_location=device))
        self.talk2dino.to(device)
        self.talk2dino.eval()

        self.clip_model, _ = clip.load("ViT-B/16", device=device, jit=False)
        self.tokenizer = clip.tokenize
        self.clip_model.eval()

    @torch.no_grad()
    def encode_text(self, texts):
        talk2dino_data = []

        for text in texts:
            text_tokens = self.tokenizer(text).to(self.device)
            text_features = self.clip_model.encode_text(text_tokens)

            dino_embed = self.talk2dino.project_clip_txt(text_features)

            talk2dino_data.append(dino_embed.squeeze().cpu().tolist())

        return torch.stack(talk2dino_data, dim=0)

    @torch.no_grad()
    def encode_image(self, x):
        B, C, H, W = x.shape

        images = x.permute(0,2,3,1).cpu().numpy()
        images = (images * 255).astype("uint8")

        inputs = self.processor(images=list(images), return_tensors="pt")
        inputs = {k:v.cuda() for k,v in inputs.items()}

        outputs = self.model(**inputs)
        features = outputs.last_hidden_state   # [B, N, C]

        patch_tokens = features[:, 5:, :]      # remove CLS

        num_patches = patch_tokens.shape[1]
        h = w = int(num_patches ** 0.5)

        patch_tokens = patch_tokens.reshape(B, h, w, -1)

        return patch_tokens
    
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
            cropped = seg_mask[y1:y2, x1:x2]

            h = y2-y1
            w = x2-x1
            long_side = max(w, h)

            cx = long_side // 2
            cy = long_side // 2
            _x1 = cx - w // 2
            _y1 = cy - h // 2
            _x2 = _x1 + w
            _y2 = _y1 + h

            seg_mask_square = torch.zeros(long_side, long_side, dtype=torch.bool).to(cropped.device)
            seg_mask_square[_y1:_y2, _x1:_x2] = cropped

            # Patch-level feature map: (D, H_patch, W_patch)
            patch_map = patch_feats[seg_id].permute(2, 0, 1)  # square size

            # Upsample to bounding box size
            seg_feats_square = F.interpolate(patch_map.unsqueeze(0), size=(long_side, long_side),
                                    mode='bilinear', align_corners=False).squeeze(0)  # (D, h_box, w_box)

            # Only write back to pixels belonging to the current segment
            dense_feature[:, seg_mask] = seg_feats_square[:, seg_mask_square]

        return dense_feature, mask
