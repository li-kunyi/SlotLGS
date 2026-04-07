import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import numpy as np
from PIL import Image
import torchvision.transforms as T

class Attention(nn.Module):
    def __init__(self, feat_dim, vl_feat_dim,
                 vl_slot_dim=512,
                 use_ins=True, use_rgb=True, use_geo=False,
                 slot_path=None):
        super().__init__()
        if use_ins:
            ins_feat_dim = feat_dim
        else:
            ins_feat_dim = 0

        if use_geo:
            self.PEn = PositionalEncoding(learnable=False, out_dim=feat_dim)
            ins_feat_dim += self.PEn.dim

        if use_rgb:
            self.rgb_embed = ColorEncoding(encode=False, out_dim=feat_dim)
            ins_feat_dim += self.rgb_embed.dim

        # Initialize slots
        ins_slot_dim = ins_feat_dim
        
        if slot_path is not None:
            self.vl_slots = np.load(slot_path)
            self.vl_slots = torch.from_numpy(self.vl_slots).cuda().float()
            num_slots = self.vl_slots.shape[0]
            vl_slot_dim = self.vl_slots.shape[-1]
            self.ins_slots = torch.randn(num_slots, ins_feat_dim, requires_grad=True, device='cuda:0').float()
            print(f"{num_slots} Slots Initialized.")
        else:
            print("Warning: No Slot Initialized! Waiting for slot loading...")
        
        # Normalization and linear layers for features
        self.compensate_in = nn.Sequential(
                nn.Linear(ins_feat_dim, 64),
                nn.ReLU(),
                nn.Linear(64, ins_slot_dim)
            )

        self.compensate_out = nn.Sequential(
            nn.Linear(ins_feat_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, vl_feat_dim)
        )
                
    def slot_init(self, ins_feat, vl_feat, tau=0.01, momentum=0.999):
        # Query, Key, Value
        q = F.normalize(self.vl_slots, dim=-1)
        k =F.normalize(vl_feat, dim=-1)  # [M, D1 + D2]
        v = ins_feat

        # Attention
        logits = torch.matmul(q, k.T)
        logits[logits < 0.7] = 0.0
        attn = F.softmax(logits / tau, dim=-1)  # [N, M]
        updates = torch.matmul(attn, v)  # [N, D]

        # Update slots with EMA
        prev = self.ins_slots
        self.ins_slots = self.ins_slots * momentum + updates * (1 - momentum)

        return F.mse_loss(self.ins_slots.detach(), prev.detach())

    
    def cross_attn(self, ins_feat):
        q = self.compensate_in(ins_feat) + ins_feat
        k = self.ins_slots
        v = self.vl_slots

        # Attention logits [N, M]
        logits = torch.matmul(q, k.T) / 0.1
        attn = F.softmax(logits, dim=-1)  # softmax over slots

        # Corss attention: vl reconstruction
        out_vl = torch.matmul(attn, v)
        vl_feat = self.compensate_out(q) + out_vl

        # Concatenate rgb and vl outputs
        output = {}
        output['vl'] = vl_feat

        return output, attn
    

    def forward(self, ins_feat):
        # Cross-Attention
        out_flat, attn = self.cross_attn(ins_feat)

        return out_flat, attn
    
    def inference(self, ins_feat, chunk_size=8192):
        N = ins_feat.shape[0]

        out_list = {}
        out_list['vl'] = []
        logit_list = []
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            chunk = ins_feat[start:end]  # [chunk, K]

            out_chunk, logit_chunk = self.cross_attn(chunk)

            out_list['vl'].append(out_chunk['vl'])
            logit_list.append(logit_chunk)

        out_flat = {}
        out_flat['vl'] = torch.cat(out_list['vl'], dim=0).to(ins_feat.device)
        logits = torch.cat(logit_list, dim=0).to(ins_feat.device)

        return out_flat, logits
    
    def get_slots(self):
        return self.ins_slots, self.vl_slots
    
    def set_slots_parameters(self, lr=0.0001):
        self.ins_slots = nn.Parameter(self.ins_slots.data, requires_grad=True)
        self.vl_slots = nn.Parameter(self.vl_slots.data, requires_grad=True)

        optimizer = torch.optim.Adam([{"params": [self.ins_slots], "lr": lr},
                                      {"params": [self.vl_slots], "lr": lr}])
        
        return optimizer
    
    def update_slots(self, ins_slots, vl_slots):
        self.ins_slots = ins_slots.detach().requires_grad_(True)
        self.vl_slots = vl_slots.detach().requires_grad_(True)

    def save(self, path):
        os.makedirs(path, exist_ok=True)

        state = self.state_dict()

        state.pop("ins_slots", None)
        state.pop("vl_slots", None)

        ckpt = {
            "model_state": state,
            "ins_slots": self.ins_slots.detach().cpu(),
            "vl_slots": self.vl_slots.detach().cpu(),
        }

        torch.save(ckpt, os.path.join(path, "attn_module.pth"))

    def load(self, path, map_location="cpu", device="cuda:0"):
        ckpt = torch.load(
            os.path.join(path, "attn_module.pth"),
            map_location=map_location
        )

        self.load_state_dict(ckpt["model_state"], strict=True)

        self.ins_slots = ckpt["ins_slots"].to(device).detach().requires_grad_(True)
        self.vl_slots = ckpt["vl_slots"].to(device).detach().requires_grad_(True)
        print(f"{self.ins_slots.shape[0]} Slots Loaded.")
    
    def load_target_feature(self, target_feature_dir, image_name, H, W, encoder='clip'):
        target_feature_name = os.path.join(target_feature_dir, image_name.split('.')[0])
        
        masks = np.load(target_feature_name + '_seg_map.npy', allow_pickle=True).item()
        seg_map = torch.from_numpy(masks['l']).cuda()  # seg_map: torch.Size([H, W]), use level 'l'
        features = torch.from_numpy(np.load(target_feature_name + '_feats.npy', allow_pickle=True)).cuda().float() # feature_map: [N, D] or [N, h, w, D] (dinov3), use level 'l'

        seg_map = F.interpolate(seg_map.unsqueeze(0).unsqueeze(0).float(), 
                                size=(H, W), mode="nearest").squeeze(0).squeeze(0).long()

        if encoder == 'dinov3':
            feature_map, valid_mask = self.get_feature_map_dinov3(seg_map, features)
        else:
            feature_map, valid_mask = self.get_feature_map(seg_map, features)
       
        return feature_map, valid_mask, seg_map
    
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
    
    @staticmethod
    def get_feature_map_dinov3(seg_map, patch_feats):
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

            h = y2-y1
            w = x2-x1
            long_side = max(w, h)

            cx = long_side // 2
            cy = long_side // 2
            _x1 = cx - w // 2
            _y1 = cy - h // 2
            _x2 = _x1 + w
            _y2 = _y1 + h

            cropped = seg_mask[y1:y2, x1:x2]
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

    
    @staticmethod
    def load_gt_image(image_path):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        img = Image.open(image_path).convert("RGB")

        transform = T.ToTensor()
        gt_image = transform(img).cuda()  # [C, H, W]，float32

        return gt_image


class PositionalEncoding(nn.Module):
    """
    Fourier Feature Positional Encoding for 3D points.
    x: tensor of shape (..., 3)
    L: number of frequency bands
    """
    def __init__(self, num_frequencies=6, include_xyz=True, learnable=False, out_dim=16):
        super().__init__()
        self.num_frequencies = num_frequencies
        self.include_xyz = include_xyz
        self.learnable = learnable

        if self.learnable:
            self.linear = nn.Sequential(
                            nn.Linear(3, 16),
                            nn.ReLU(),
                            nn.Linear(16, out_dim)
                        )
        
            self.dim = out_dim
        else:
            self.dim = 3 * 2 * num_frequencies + 3 if self.include_xyz else 3 * 2 * num_frequencies
            # [2^0, 2^1, ..., 2^(L-1)]
            self.freq_bands = 2.0 ** torch.arange(num_frequencies)

    def forward(self, x):
        """
        x: (..., 3) 3D coordinates
        returns: (..., 3*2*num_frequencies)
        """
        if self.learnable:
            C = x.shape[-1]

            x = x.view(-1, C)    # [H*W, C]
            out = self.linear(x)           # [H*W, D]
        else:
            out = [x] if self.include_xyz else []
            for freq in self.freq_bands:
                out.append(torch.sin(freq * x))
                out.append(torch.cos(freq * x))
            out = torch.cat(out, dim=-1)

        return out

class ColorEncoding(nn.Module):
    """
    Fourier Feature Positional Encoding for 3D points.
    x: tensor of shape (..., 3)
    L: number of frequency bands
    """
    def __init__(self, encode=True, out_dim=16):
        super().__init__()
        self.encode = encode

        if self.encode:
            self.linear = nn.Linear(3, out_dim)
            self.dim = out_dim
        else:
            self.dim = 3
        
    def forward(self, x):
        if self.encode:
            C = x.shape[-1]

            x = x.view(-1, C)    # [H*W, C]
            out = self.linear(x)           # [H*W, D]
        else:
            out = x
        return out