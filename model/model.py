import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import numpy as np
from PIL import Image
import torchvision.transforms as T


class Attention(nn.Module):
    def __init__(self, feat_dim, vl_feat_dim, num_slots, hidden_dim, vl_slot_dim, iters=3, 
                 use_ins=True, use_rgb=True, use_geo=False, slot_path=None, random_init=False):
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

        if not random_init and slot_path is not None:
            slots = np.load(slot_path)
            slots = torch.from_numpy(slots).cuda().float()
            self.vl_slots = slots.requires_grad_(True)
            num_slots, vl_slot_dim = self.vl_slots.shape
            print(f"{num_slots} Slots Initialized from Dataset.")
        elif random_init:
            self.vl_slots = torch.randn(num_slots, vl_slot_dim).cuda().requires_grad_(True)
            self.vl_slots = F.normalize(self.vl_slots, dim=-1)
            print(f"{num_slots} Slots Initialized Randomly.")
        
        # Normalization and linear layers
        self.norm_app_feat = nn.LayerNorm(app_feat_dim)
        self.norm_vl_feat = nn.LayerNorm(vl_feat_dim)
        self.norm_vl_slots = nn.LayerNorm(vl_slot_dim)

        self.proj_q = nn.Linear(app_feat_dim, hidden_dim)
        self.proj_k = nn.Linear(vl_slot_dim, hidden_dim)

        # Residual linear layers
        self.residual_connection = nn.Sequential(
                nn.Linear(hidden_dim, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, vl_feat_dim)
            )
    
    def cross_attn(self, app_feat, alpha=1.0):
        q = self.proj_q(self.norm_app_feat(app_feat))
        k = self.proj_k(self.norm_vl_slots(self.vl_slots))
        v = F.normalize(self.vl_slots, dim=-1)

        M, D = k.shape

        # Attention logits [N, M]
        logits = torch.matmul(q, k.T) / math.sqrt(D)
        attn = F.softmax(logits, dim=-1)  # softmax over slots

        # Corss attention: vl reconstruction
        out_vl = torch.matmul(attn, v)
        vl_feat = alpha * out_vl + (1 - alpha) * self.residual_connection(q) 
        # vl_feat = F.normalize(vl_feat, dim=-1)

        # Concatenate rgb and vl outputs
        output = {}
        output['vl'] = vl_feat

        return output, attn

    def forward(self, app_feat):
        # Cross-Attention
        out_flat, attn = self.cross_attn(app_feat)

        return out_flat, attn
    
    def inference(self, app_feat, chunk_size=8192):
        N = app_feat.shape[0]

        out_list = {}
        out_list['vl'] = []
        logit_list = []
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            chunk = app_feat[start:end]  # [chunk, K]

            out_chunk, logit_chunk = self.cross_attn(chunk)

            out_list['vl'].append(out_chunk['vl'])
            logit_list.append(logit_chunk)

        out_flat = {}
        out_flat['vl'] = torch.cat(out_list['vl'], dim=0).to(app_feat.device)
        logits = torch.cat(logit_list, dim=0).to(app_feat.device)

        return out_flat, logits

    def get_slots(self):
        return self.vl_slots

    def set_slots_optimizer(self, lr=1e-3):
        self.vl_slots = nn.Parameter(self.vl_slots.detach().clone(), requires_grad=True)
        optimizer = torch.optim.Adam([{"params": [self.vl_slots], "lr": lr}])
        return optimizer
            
    def save(self, path):
        os.makedirs(path, exist_ok=True)

        state = self.state_dict()
        state.pop("vl_slots", None)

        ckpt = {
            "model_state": state,
            "vl_slots": self.vl_slots.detach().cpu(),
        }

        torch.save(ckpt, os.path.join(path, "attn_module.pth"))

    def load(self, path, map_location="cpu", device="cuda:0"):
        ckpt = torch.load(
            os.path.join(path, "attn_module.pth"),
            map_location=map_location
        )

        self.load_state_dict(ckpt["model_state"], strict=True)

        self.vl_slots = ckpt["vl_slots"].to(device).detach().requires_grad_(True)
        print(f"{self.vl_slots.shape[0]} Slots Loaded.")

    
    def load_target_feature(self, target_feature_dir, image_name, H, W, encoder='clip', level='l'):
        target_feature_name = os.path.join(target_feature_dir, image_name.split('.')[0])
        
        masks = np.load(target_feature_name + '_seg_map.npy', allow_pickle=True).item()
        seg_map = torch.from_numpy(masks[level]).cuda()  # seg_map: torch.Size([H, W]), use level 'l'
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


def get_encoder(encoding, input_dim=3,
                degree=4, n_bins=16, n_frequencies=12,
                n_levels=16, level_dim=2, 
                base_resolution=16, log2_hashmap_size=19, 
                desired_resolution=512):
    import tinycudann as tcnn
    
    # Dense grid encoding
    if 'dense' in encoding.lower():
        n_levels = 4
        per_level_scale = np.exp2(np.log2(desired_resolution  / base_resolution) / (n_levels - 1))
        embed = tcnn.Encoding(
            n_input_dims=input_dim,
            encoding_config={
                    "otype": "Grid",
                    "type": "Dense",
                    "n_levels": n_levels,
                    "n_features_per_level": level_dim,
                    "base_resolution": base_resolution,
                    "per_level_scale": per_level_scale,
                    "interpolation": "Linear"},
                dtype=torch.float
        )
        out_dim = embed.n_output_dims
    
    # Sparse grid encoding
    elif 'hash' in encoding.lower() or 'tiled' in encoding.lower():
        print('Hash size', log2_hashmap_size)
        per_level_scale = np.exp2(np.log2(desired_resolution  / base_resolution) / (n_levels - 1))
        embed = tcnn.Encoding(
            n_input_dims=input_dim,
            encoding_config={
                "otype": 'HashGrid',
                "n_levels": n_levels,
                "n_features_per_level": level_dim,
                "log2_hashmap_size": log2_hashmap_size,
                "base_resolution": base_resolution,
                "per_level_scale": per_level_scale
            },
            dtype=torch.float
        )
        out_dim = embed.n_output_dims

    # Spherical harmonics encoding
    elif 'spherical' in encoding.lower():
        embed = tcnn.Encoding(
                n_input_dims=input_dim,
                encoding_config={
                "otype": "SphericalHarmonics",
                "degree": degree,
                },
                dtype=torch.float
            )
        out_dim = embed.n_output_dims
    
    # OneBlob encoding
    elif 'blob' in encoding.lower():
        print('Use blob')
        embed = tcnn.Encoding(
                n_input_dims=input_dim,
                encoding_config={
                "otype": "OneBlob", #Component type.
	            "n_bins": n_bins
                },
                dtype=torch.float
            )
        out_dim = embed.n_output_dims
    
    # Frequency encoding
    elif 'freq' in encoding.lower():
        print('Use frequency')
        embed = tcnn.Encoding(
                n_input_dims=input_dim,
                encoding_config={
                "otype": "Frequency", 
                "n_frequencies": n_frequencies
                },
                dtype=torch.float
            )
        out_dim = embed.n_output_dims
    
    # Identity encoding
    elif 'identity' in encoding.lower():
        embed = tcnn.Encoding(
                n_input_dims=input_dim,
                encoding_config={
                "otype": "Identity"
                },
                dtype=torch.float
            )
        out_dim = embed.n_output_dims

    return embed, out_dim


class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(x + self.net(x))


class MLP(nn.Module):
    def __init__(self, n_input_dims, n_output_dims, hidden_dim, num_blocks=2):
        super().__init__()

        # input projection
        self.input = nn.Sequential(
            nn.Linear(n_input_dims, hidden_dim),
            nn.ReLU()
        )

        # residual blocks
        # self.blocks = nn.Sequential(*[
        #     ResBlock(hidden_dim) for _ in range(num_blocks)
        # ])

        self.blocks = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # output head
        self.output = nn.Linear(hidden_dim, n_output_dims)

    def forward(self, x):
        x = self.input(x)
        x = self.blocks(x)
        return self.output(x)
    

class HashInstanceField(nn.Module):
    def __init__(self, output_dims=16, hidden_dim=128, 
                 hash_size=16, resolution=256, num_layers=5):
        super().__init__()
        import tinycudann as tcnn

        self.grid_fn, self.grid_dim = get_encoder('HashGrid', 
                                                  log2_hashmap_size=hash_size, 
                                                  desired_resolution=resolution)
        n_input_dims = self.grid_dim

        self.pe_fn, self.pe_dim = get_encoder('OneBlob', n_bins=16)
        n_input_dims += self.pe_dim
        
        self.decoder = tcnn.Network(n_input_dims=n_input_dims + 3,  # Add 3 for the additional RGB input
                                    n_output_dims=output_dims,
                                    network_config={
                                        "otype": "FullyFusedMLP", # use CutlassMLP if not support FullyFusedMLP
                                        "activation": "ReLU",
                                        "output_activation": "None",
                                        "n_neurons": hidden_dim,
                                        "n_hidden_layers": num_layers})
        
        # self.decoder = MLP(n_input_dims, n_output_dims, hidden_dim)

    def normalization(self, x):
        return torch.sigmoid(x)
        
    def forward(self, x, batch=50000):
        num = x.shape[0]
        out = []
        for i in range(num // batch + 1):
            start = i * batch
            end = min((i + 1) * batch, num)
            _x = x[start:end]

            if end - start > 0:
                _p = self.normalization(_x[:, :3])
                _grid = self.grid_fn(_p)

                _pe = self.pe_fn(_p)
                y = self.decoder(torch.cat((_pe, _grid, _x[:, 3:]), dim=-1))
                out.append(y)

        out = torch.cat(out, dim=0).cuda().float()
        return out
    

class FourierInstanceField(nn.Module):
    def __init__(self, output_dims=16, hidden_dim=128):
        super().__init__()

        self.pe_fn = PositionalEncoding(learnable=False, num_frequencies=5)
        input_dims = self.pe_fn.dim
        
        self.decoder = MLP(input_dims + 3, output_dims, hidden_dim)  # Add 3 for the additional RGB input
        
    def forward(self, x, batch=50000):
        num = x.shape[0]
        out = []
        for i in range(num // batch + 1):
            start = i * batch
            end = min((i + 1) * batch, num)
            _x = x[start:end]

            if end - start > 0:
                _pe = self.pe_fn(_x[:, :3])
                y = self.decoder(torch.cat((_pe, _x[:, 3:]), dim=-1))
                out.append(y)

        out = torch.cat(out, dim=0).cuda().float()
        return out


class ViewCompensate(nn.Module):
    def __init__(self, input_dims, output_dims=16, hidden_dim=128, num_layers=5, pos_edb=True):
        super().__init__()
        self.pos_edb = pos_edb
        if self.pos_edb:
            self.pe_fn = PositionalEncoding(learnable=False, num_frequencies=5)
            input_dims += self.pe_fn.dim
        else:
            input_dims += 3
        
        self.decoder = nn.Linear(input_dims, output_dims)
        
    def forward(self, x, pos):
        if self.pos_edb:
            pos = self.pe_fn(pos)
        out = self.decoder(torch.cat((x, pos), dim=-1))
        return out