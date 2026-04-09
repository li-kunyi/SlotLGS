import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import numpy as np
from PIL import Image
import torchvision.transforms as T

class Attention(nn.Module):
    def __init__(self, feat_dim, vl_feat_dim, num_slots, app_slot_dim, vl_slot_dim, iters=3, use_ins=True, use_rgb=True, use_geo=False):
        super().__init__()
        self.slot_iters = iters
        self.num_slots = num_slots
        self.avg_attn_mass = torch.zeros(num_slots, device='cuda:0')
        self.attn_count = 0
        self.attn_max = torch.zeros(num_slots, device='cuda:0')
        self.slot_ent = torch.zeros(num_slots, device='cuda:0')

        if use_ins:
            app_feat_dim = feat_dim
        else:
            app_feat_dim = 0

        if use_geo:
            self.PEn = PositionalEncoding(learnable=False, out_dim=feat_dim)
            app_feat_dim += self.PEn.dim

        if use_rgb:
            self.rgb_embed = ColorEncoding(encode=False, out_dim=feat_dim)
            app_feat_dim += self.rgb_embed.dim
        
        # Initialize slots
        self.app_slots = torch.randn(num_slots, app_slot_dim, requires_grad=True, device='cuda:0')
        self.vl_slots = torch.randn(num_slots, vl_slot_dim, requires_grad=True, device='cuda:0')

        # Normalization and linear layers for features
        self.norm_app = nn.LayerNorm(app_feat_dim)
        self.linear_app = nn.Linear(app_feat_dim, app_slot_dim)  

        self.norm_vl = nn.LayerNorm(vl_feat_dim)
        self.linear_vl = nn.Linear(vl_feat_dim, vl_slot_dim)

        # Normalization and linear layers for slots
        self.norm_app_slots = nn.LayerNorm(app_slot_dim)
        self.linear_app_slots = nn.Linear(app_slot_dim, app_slot_dim)

        self.norm_vl_slots = nn.LayerNorm(vl_slot_dim)
        self.linear_vl_slots = nn.Linear(vl_slot_dim, vl_slot_dim)     

        # Residual linear layers
        self.linear_residual = nn.Linear(app_feat_dim, vl_slot_dim)  

        # GRU cells for slot updates
        self.gru_app = nn.GRUCell(app_slot_dim, app_slot_dim)
        self.gru_vl = nn.GRUCell(vl_slot_dim, vl_slot_dim)
        
        self.ln_vl = nn.LayerNorm(vl_slot_dim)
        self.ln_rgb_ins = nn.LayerNorm(app_slot_dim)

        self.mlp_rgb = nn.Sequential(
            nn.Linear(app_slot_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

        self.mlp_ins = nn.Sequential(
            nn.Linear(app_slot_dim, 64),
            nn.ReLU(),
            nn.Linear(64, feat_dim)
        )

        self.mlp_vl = nn.Sequential(
            nn.Linear(vl_slot_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, vl_feat_dim)
        )
                
    def slot_attn(self, app_feat, vl_feat, app_slots, vl_slots):
        # slots as queries
        query_app = self.linear_app_slots(self.norm_app_slots(app_slots))  # [N, D1]
        query_vl = self.linear_vl_slots(self.norm_vl_slots(vl_slots))  # [N, D2]

        # features as keys
        key_app = self.linear_app(self.norm_app(app_feat))  # [M, D1]
        key_vl = self.linear_vl(self.norm_vl(vl_feat))  # [M, D2] 

        D1 = query_app.shape[-1]  # app_slot_dim
        D2 = query_vl.shape[-1]  # vl_slot_dim
        D = D1 + D2

        # Query, Key, Value
        # q = torch.cat([query_app, query_vl], dim=-1)  # [N, D1 + D2]
        # k = torch.cat([key_app, key_vl], dim=-1)  # [M, D1 + D2]
        # v = k

        # q = query_app  # [N, D1]
        # k = key_app  # [M, D1]
        # v = torch.cat([key_app, key_vl], dim=-1)  # [M, D1 + D2]

        # Attention
        # logits = torch.matmul(q, k.T) / math.sqrt(D)
        # attn = F.softmax(logits, dim=-1)  # [N, M]
        # updates = torch.matmul(attn, v)  # [N, D]
        # updates_in = updates[:, :D1]
        # updates_tgt = updates[:, D1:]

        sim_app = torch.matmul(query_app, key_app.T) / math.sqrt(D1)
        sim_vl = torch.matmul(query_vl, key_vl.T) / math.sqrt(D2)
        logits = (sim_app + sim_vl) / 2
        attn = F.softmax(logits, dim=-1)  # [N, M]

        updates_in = torch.matmul(attn, key_app)  # [N, D1]
        updates_tgt = torch.matmul(attn, key_vl)  # [N, D2]

        # GRU update
        updated_app_slots = self.gru_app(updates_in, app_slots)
        updated_vl_slots = self.gru_vl(updates_tgt, vl_slots)

        # updated_app_slots = updates_in
        # updated_vl_slots = updates_tgt

        return updated_app_slots, updated_vl_slots
    
    def cross_attn(self, app_feat, app_slots, vl_slots):
        q = self.linear_app(self.norm_app(app_feat))
        k = self.linear_app_slots(self.norm_app_slots(app_slots))
        v = self.linear_vl_slots(self.norm_vl_slots(vl_slots))

        res = self.linear_residual(self.norm_app(app_feat))

        M, D = k.shape

        # Attention logits [N, M]
        logits = torch.matmul(q, k.T) / math.sqrt(D)
        attn = F.softmax(logits, dim=-1)  # softmax over slots

        # Corss attention: vl reconstruction
        out_vl = torch.matmul(attn, v) + res
        vl_feat = self.mlp_vl(self.ln_vl(out_vl)) 
        
        # out_vl = torch.matmul(attn, v)
        # vl_feat = self.mlp_vl(out_vl)

        # Self attention: apperance reconstruction
        out_app = torch.matmul(attn, k) + q
        out_app_norm = self.ln_rgb_ins(out_app)
        
        rgb = self.mlp_rgb(out_app_norm)
        ins_feat = self.mlp_ins(out_app_norm)

        # out_app = torch.matmul(attn, k)
        # rgb = self.mlp_rgb(out_app + q)
        # ins_feat = self.mlp_ins(out_app)

        # Concatenate rgb and vl outputs
        output = {}
        output['rgb'] = rgb
        output['ins'] = ins_feat
        output['vl'] = vl_feat

        return output, attn

    def forward(self, app_feat, vl_feat, momentum=0.999):
        # Slot Attention -> update slots
        app_slots_updates, vl_slots_updates = self.slot_attn(app_feat, vl_feat, self.app_slots, self.vl_slots)

        # Update slots with EMA
        updated_app_slots = self.app_slots * momentum + app_slots_updates * (1 - momentum)
        updated_vl_slots = self.vl_slots * momentum + vl_slots_updates * (1 - momentum)

        # Cross-Attention
        out_flat, attn = self.cross_attn(app_feat, updated_app_slots, updated_vl_slots)

        return out_flat, updated_app_slots, updated_vl_slots, attn
    
    def inference(self, app_feat, chunk_size=8192):
        N = app_feat.shape[0]

        out_list = {}
        out_list['rgb'] = []
        out_list['ins'] = []
        out_list['vl'] = []
        logit_list = []
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            chunk = app_feat[start:end]  # [chunk, K]

            out_chunk, logit_chunk = self.cross_attn(chunk, self.app_slots, self.vl_slots)

            out_list['rgb'].append(out_chunk['rgb'])
            out_list['ins'].append(out_chunk['ins'])
            out_list['vl'].append(out_chunk['vl'])
            logit_list.append(logit_chunk)

        out_flat = {}
        out_flat['rgb'] = torch.cat(out_list['rgb'], dim=0).to(app_feat.device)
        out_flat['ins'] = torch.cat(out_list['ins'], dim=0).to(app_feat.device)
        out_flat['vl'] = torch.cat(out_list['vl'], dim=0).to(app_feat.device)
        logits = torch.cat(logit_list, dim=0).to(app_feat.device)

        return out_flat, logits

    def get_slot_logits(self, app_feat, app_slots):
        q = self.linear_app(self.norm_app(app_feat))
        k = self.linear_app_slots(self.norm_app_slots(app_slots))

        M, D = k.shape

        # Attention logits [N, M]
        logits = torch.matmul(q, k.T) / math.sqrt(D)

        return logits
    
    def get_slots(self):
        return self.app_slots, self.vl_slots
    
    def update_slots(self, app_slots, vl_slots):
        self.app_slots = app_slots.detach().requires_grad_(True)
        self.vl_slots = vl_slots.detach().requires_grad_(True)

    def add_attn_status(self, weights, slot_ent):
        # for pruning
        self.avg_attn_mass += weights.mean(dim=0)
        self.attn_count += 1

        self.slot_ent += slot_ent

        weight_max = torch.max(weights, dim=0).values
        self.attn_max = torch.max(weight_max, self.attn_max)
            
    def densification_and_prune(self, mass_th=0.01, max_th=0.9, slot_ent_th=0.2, prune=True, densify=True, momentum=0.):
        num_slots = self.app_slots.shape[0]
        avg_attn_mass = self.avg_attn_mass / self.attn_count
        print(f"Number of Slots, Before: {num_slots}")

        # Minimum slots: 16
        if num_slots >= 16 and prune:
            # Prune
            mass_valid_mask = (avg_attn_mass > mass_th)
            max_valid_mask = (self.attn_max > max_th)
            valid_mask = torch.logical_and(mass_valid_mask, max_valid_mask)

            if valid_mask.sum() < 16:
                _, valid_mask = torch.topk(avg_attn_mass, k=16, largest=True)
            
            self.app_slots = self.app_slots[valid_mask]
            self.vl_slots = self.vl_slots[valid_mask]
            self.slot_ent = self.slot_ent[valid_mask]

            avg_attn_mass = avg_attn_mass[valid_mask]

        # Maximum slots: 128
        num_slots = self.app_slots.shape[0]
        if num_slots >= 128 and densify:
            _, valid_mask = torch.topk(avg_attn_mass, k=128, largest=True)
            
            self.app_slots = self.app_slots[valid_mask]
            self.vl_slots = self.vl_slots[valid_mask]

        elif num_slots < 128 and densify:
            # Densify
            avg_slot_ent = self.slot_ent / self.attn_count
            valid_mask = (avg_slot_ent > slot_ent_th)

            # _, valid_mask = torch.topk(avg_attn_mass, k=6, largest=True)

            new_in_slots = self.app_slots[valid_mask]
            new_tgt_slots = self.vl_slots[valid_mask]
            
            new_num, app_slot_dim = new_in_slots.shape
            new_num, vl_slot_dim = new_tgt_slots.shape

            new_in_slots = momentum * new_in_slots + (1 - momentum) * torch.randn(new_num, app_slot_dim, requires_grad=True, device='cuda:0')
            new_tgt_slots = momentum * new_tgt_slots + (1 - momentum) * torch.randn(new_num, vl_slot_dim, requires_grad=True, device='cuda:0')

            self.app_slots = momentum * self.app_slots + (1 - momentum) * torch.randn(num_slots, app_slot_dim, requires_grad=True, device='cuda:0')
            self.vl_slots = momentum * self.vl_slots + (1 - momentum) * torch.randn(num_slots, vl_slot_dim, requires_grad=True, device='cuda:0')

            self.app_slots = torch.cat([self.app_slots, new_in_slots], dim=0)
            self.vl_slots = torch.cat([self.vl_slots, new_tgt_slots], dim=0)

        # Reset status
        self.num_slots = self.app_slots.shape[0]

        self.avg_attn_mass = torch.zeros(self.num_slots, device='cuda:0')
        self.attn_count = 0
        self.attn_max = torch.zeros(self.num_slots, device='cuda:0')
        self.slot_ent = torch.zeros(self.num_slots, device='cuda:0')

        print(f"Number of Slots, After: {self.num_slots}")

    def save(self, path):
        os.makedirs(path, exist_ok=True)

        ckpt = {
            "model_state": self.state_dict(),
            "app_slots": self.app_slots.detach().cpu(),
            "vl_slots": self.vl_slots.detach().cpu(),
        }

        torch.save(ckpt, os.path.join(path, "attn_module.pth"))

    def load(self, path, map_location="cpu", device="cuda:0"):
        ckpt = torch.load(
            os.path.join(path, "attn_module.pth"),
            map_location=map_location
        )

        self.load_state_dict(ckpt["model_state"], strict=True)

        self.app_slots = ckpt["app_slots"].to(device).detach().requires_grad_(True)
        self.vl_slots = ckpt["vl_slots"].to(device).detach().requires_grad_(True)
    
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
    def __init__(self, num_frequencies=4, include_xyz=True, learnable=False, out_dim=16):
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

        self.norm = nn.LayerNorm(self.dim)

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

        return self.norm(out)

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