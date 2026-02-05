import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import numpy as np
from PIL import Image
import torchvision.transforms as T

class Attention(nn.Module):
    def __init__(self, in_feat_dim, tgt_feat_dim, num_slots, in_slot_dim, tgt_slot_dim, iters=3, use_geo=False):
        super().__init__()
        self.slot_iters = iters
        self.num_slots = num_slots
        self.avg_attn_mass = torch.zeros(num_slots, device='cuda:0')
        self.attn_count = 0
        self.attn_max = torch.zeros(num_slots, device='cuda:0')
        self.densify_count = torch.zeros(num_slots, device='cuda:0')

        self.use_geo = use_geo
        if self.use_geo:
            self.PEn = PositionalEncoding(learnable=True, out_dim=16)
            in_feat_dim += self.PEn.dim
        
        # Initialize slots
        self.in_slots = torch.randn(num_slots, in_slot_dim, requires_grad=True, device='cuda:0')
        self.tgt_slots = torch.randn(num_slots, tgt_slot_dim, requires_grad=True, device='cuda:0')

        # Normalization and linear layers for features
        self.norm_input = nn.LayerNorm(in_feat_dim)
        self.linear_input = nn.Linear(in_feat_dim, in_slot_dim)  

        self.norm_tgt = nn.LayerNorm(tgt_feat_dim)
        self.linear_tgt = nn.Linear(tgt_feat_dim, tgt_slot_dim)

        # Normalization and linear layers for slots
        self.norm_in_slots = nn.LayerNorm(in_slot_dim)
        self.linear_in_slots = nn.Linear(in_slot_dim, in_slot_dim)

        self.norm_tgt_slots = nn.LayerNorm(tgt_slot_dim)
        self.linear_tgt_slots = nn.Linear(tgt_slot_dim, tgt_slot_dim)      

        # Residual linear layers
        self.linear_residual = nn.Linear(in_feat_dim, tgt_slot_dim)  

        # GRU cells for slot updates
        self.gru_in = nn.GRUCell(in_slot_dim, in_slot_dim)
        self.gru_tgt = nn.GRUCell(tgt_slot_dim, tgt_slot_dim)
        

        self.ln_semantic = nn.LayerNorm(tgt_slot_dim)
        self.ln_rgb = nn.LayerNorm(in_slot_dim)

        self.mlp_rgb = nn.Sequential(
            nn.Linear(in_slot_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, in_feat_dim)
        )

        self.mlp_semantic = nn.Sequential(
            nn.Linear(tgt_slot_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, tgt_feat_dim)
        )
        
                
    def slot_attn(self, inputs, targets, in_slots, tgt_slots):
        # slots as queries
        query_input = self.linear_in_slots(self.norm_in_slots(in_slots))  # [N, D1]
        query_tgt = self.linear_tgt_slots(self.norm_tgt_slots(tgt_slots))  # [N, D2]

        # features as keys
        key_input = self.linear_input(self.norm_input(inputs))  # [M, D1]
        key_tgt = self.linear_tgt(self.norm_tgt(targets))  # [M, D2] 

        D1 = query_input.shape[-1]  # in_slot_dim
        D2 = query_tgt.shape[-1]  # tgt_slot_dim
        D = D1 + D2

        # Query, Key, Value
        q = torch.cat([query_input, query_tgt], dim=-1)  # [N, D1 + D2]
        k = torch.cat([key_input, key_tgt], dim=-1)  # [M, D1 + D2]
        v = k

        # Attention
        logits = torch.matmul(q, k.T) / math.sqrt(D)
        attn = F.softmax(logits, dim=-1)  # [N, M]
        updates = torch.matmul(attn, v)  # [N, D]

        updates_in = updates[:, :D1]
        updates_tgt = updates[:, D1:]

        # GRU update
        updated_in_slots = self.gru_in(updates_in, in_slots)
        updated_tgt_slots = self.gru_tgt(updates_tgt, tgt_slots)

        return updated_in_slots, updated_tgt_slots
    
    def cross_attn(self, inputs, in_slots, tgt_slots):
        q = self.linear_input(self.norm_input(inputs))
        k = self.linear_in_slots(self.norm_in_slots(in_slots))
        v = self.linear_tgt_slots(self.norm_tgt_slots(tgt_slots))

        res = self.linear_residual(self.norm_input(inputs))

        M, D = k.shape

        # Attention logits [N, M]
        logits = torch.matmul(q, k.T) / math.sqrt(D)
        attn = F.softmax(logits, dim=-1)  # softmax over slots

        # Corss attention: semantic reconstruction
        out_semantics = torch.matmul(attn, v) + res
        semantics = self.mlp_semantic(self.ln_semantic(out_semantics)) 
        semantics = F.normalize(semantics)

        # Self attention: apperance reconstruction
        out_rgbs = torch.matmul(attn, k) + q
        rgbs = self.mlp_rgb(self.ln_rgb(out_rgbs))

        # Concatenate rgb and semantic outputs
        output = torch.cat([rgbs, semantics], dim=-1)

        return output, attn

    def forward(self, in_flat, tgt_flat, momentum=0.995):
        # Slot Attention -> update slots
        in_slots_updates, tgt_slots_updates = self.slot_attn(in_flat, tgt_flat, self.in_slots, self.tgt_slots)

        # Update slots with EMA
        updated_in_slots = self.in_slots * momentum + in_slots_updates * (1 - momentum)
        updated_tgt_slots = self.tgt_slots * momentum + tgt_slots_updates * (1 - momentum)

        # Cross-Attention
        out_flat, attn = self.cross_attn(in_flat, updated_in_slots, updated_tgt_slots)

        return out_flat, updated_in_slots, updated_tgt_slots, attn
    
    def inference(self, in_flat):
        out_flat, logits = self.cross_attn(in_flat, self.in_slots, self.tgt_slots)
        return out_flat, logits
    
    def get_logits(self, inputs, in_slots):
        q = self.linear_input(self.norm_input(inputs))
        k = self.linear_in_slots(self.norm_in_slots(in_slots))
        M, D = k.shape

        # Attention logits [N, M]
        logits = torch.matmul(q, k.T) / math.sqrt(D)
        return logits
    
    def get_slots(self):
        return self.in_slots, self.tgt_slots
    
    def update_slots(self, in_slots, tgt_slots):
        self.in_slots = in_slots.detach().requires_grad_(True)
        self.tgt_slots = tgt_slots.detach().requires_grad_(True)

    def add_attn_status(self, weights):
        # for pruning
        self.avg_attn_mass += weights.mean(dim=0)
        self.attn_count += 1

        weight_max = torch.max(weights, dim=0).values
        self.attn_max = torch.max(weight_max, self.attn_max)
            
    def densification_and_prune(self, mass_th=0.02, max_th=0.9, prune=True, densify=True, momentum=0.7):
        num_slots = self.in_slots.shape[0]
        avg_attn_mass = self.avg_attn_mass / self.attn_count
        print(f"Number of Slots, Before: {num_slots}")

        # Minimum slots: 16
        if num_slots >= 16 and prune:
            # Prune
            mass_valid_mask = (avg_attn_mass > mass_th)
            max_valid_mask = (self.attn_max > max_th)
            valid_mask = torch.logical_and(mass_valid_mask, max_valid_mask)

            if valid_mask.sum() < 8:
                _, valid_mask = torch.topk(avg_attn_mass, k=8, largest=True)
            
            self.in_slots = self.in_slots[valid_mask]
            self.tgt_slots = self.tgt_slots[valid_mask]
            self.densify_count = self.densify_count[valid_mask]

            avg_attn_mass = avg_attn_mass[valid_mask]

        # Maximum slots: 128
        num_slots = self.in_slots.shape[0]
        if num_slots >= 128 and densify:
            _, top_indices = torch.topk(avg_attn_mass, k=128, largest=True)
            
            self.in_slots = self.in_slots[top_indices]
            self.tgt_slots = self.tgt_slots[top_indices]

        elif num_slots < 128 and densify:
            # Densify
            _, top_indices = torch.topk(avg_attn_mass, k=6, largest=True)
            new_in_slots = self.in_slots[top_indices]
            new_tgt_slots = self.tgt_slots[top_indices]
            
            new_num, in_slot_dim = new_in_slots.shape
            new_num, tgt_slot_dim = new_tgt_slots.shape

            new_in_slots = momentum * new_in_slots + (1 - momentum) * torch.randn(new_num, in_slot_dim, requires_grad=True, device='cuda:0')
            new_tgt_slots = momentum * new_tgt_slots + (1 - momentum) * torch.randn(new_num, tgt_slot_dim, requires_grad=True, device='cuda:0')

            random_in_slots = torch.randn(2, in_slot_dim, requires_grad=True, device='cuda:0')
            random_tgt_slots = torch.randn(2, tgt_slot_dim, requires_grad=True, device='cuda:0')

            self.in_slots[top_indices] = momentum * self.in_slots[top_indices] + (1 - momentum) * torch.randn(new_num, in_slot_dim, requires_grad=True, device='cuda:0')
            self.tgt_slots[top_indices] = momentum * self.tgt_slots[top_indices] + (1 - momentum) * torch.randn(new_num, tgt_slot_dim, requires_grad=True, device='cuda:0')

            self.in_slots = torch.cat([self.in_slots, new_in_slots, random_in_slots], dim=0)
            self.tgt_slots = torch.cat([self.tgt_slots, new_tgt_slots, random_tgt_slots], dim=0)

        # Reset status
        self.num_slots = self.in_slots.shape[0]

        self.avg_attn_mass = torch.zeros(self.num_slots, device='cuda:0')
        self.attn_count = 0
        self.attn_max = torch.zeros(self.num_slots, device='cuda:0')
        self.densify_count = torch.zeros(self.num_slots, device='cuda:0')

        print(f"Number of Slots, After: {self.num_slots}")

    def save(self, path):
        os.makedirs(path, exist_ok=True)

        ckpt = {
            "model_state": self.state_dict(),
            "in_slots": self.in_slots.detach().cpu(),
            "tgt_slots": self.tgt_slots.detach().cpu(),
        }

        torch.save(ckpt, os.path.join(path, "attn_module.pth"))

    def load(self, path, map_location="cpu", device="cuda:0"):
        ckpt = torch.load(
            os.path.join(path, "attn_module.pth"),
            map_location=map_location
        )

        self.load_state_dict(ckpt["model_state"], strict=True)

        self.in_slots = ckpt["in_slots"].to(device).detach().requires_grad_(True)
        self.tgt_slots = ckpt["tgt_slots"].to(device).detach().requires_grad_(True)

    def load_gt_image(self, image_path):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        img = Image.open(image_path).convert("RGB")

        transform = T.ToTensor()
        gt_image = transform(img).cuda()  # [C, H, W]，float32

        return gt_image

    def get_instance_masks(self, instance_mask_dir, image_name):
        base_dir, instance_name = '/'.join(instance_mask_dir.split('/')[:-1]), instance_mask_dir.split('/')[-1]

        if os.path.exists(os.path.join(base_dir, 'train', instance_name, image_name+ '.npy')):
            instance_mask_name = os.path.join(base_dir, 'train', instance_name, image_name)
        elif os.path.exists(os.path.join(base_dir, 'test', instance_name, image_name+ '.npy')):
            instance_mask_name = os.path.join(base_dir, 'test', instance_name, image_name)
        else: 
            instance_mask_name = os.path.join(instance_mask_dir, image_name)

        instance_masks = torch.from_numpy(np.load(instance_mask_name + ".npy"))
        return instance_masks.cuda()
        
    
    def load_target_feature(self, target_feature_dir, image_name, feature_level):
        
        target_feature_name = os.path.join(target_feature_dir, image_name.split('.')[0])
        
        seg_map = torch.from_numpy(np.load(target_feature_name + '_s.npy'))  # seg_map: torch.Size([4, H, W])
        if seg_map.ndim == 2:
            seg_map = seg_map.unsqueeze(0)
        feature_map = torch.from_numpy(np.load(target_feature_name + '_f.npy')) # feature_map: torch.Size([N, 512])
        seg_map = seg_map.cuda()
        feature_map = feature_map.cuda()

        _, H, W = seg_map.shape

        y, x = torch.meshgrid(torch.arange(0, H, device='cuda'), torch.arange(0, W, device='cuda'))
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)

        seg = seg_map[..., y, x].squeeze(-1).long()
        mask = seg != -1
        if feature_level == 0: # default
            point_feature1 = feature_map[seg[0:1]].squeeze(0)
            mask = mask[0:1].reshape(1, H, W)
        elif feature_level == 1: # s
            point_feature1 = feature_map[seg[1:2]].squeeze(0)
            mask = mask[1:2].reshape(1, H, W)
        elif feature_level == 2: # m
            point_feature1 = feature_map[seg[2:3]].squeeze(0)
            mask = mask[2:3].reshape(1, H, W)
        elif feature_level == 3: # l
            point_feature1 = feature_map[seg[3:4]].squeeze(0)
            mask = mask[3:4].reshape(1, H, W)
        else:
            raise ValueError("feature_level=", feature_level)
        
        point_feature = point_feature1.reshape(H, W, -1).permute(2, 0, 1)
       
        return point_feature, mask


class PositionalEncoding(nn.Module):
    """
    Fourier Feature Positional Encoding for 3D points.
    x: tensor of shape (..., 3)
    L: number of frequency bands
    """
    def __init__(self, num_frequencies=10, include_xyz=True, learnable=False, out_dim=16):
        super().__init__()
        self.num_frequencies = num_frequencies
        self.include_xyz = include_xyz
        self.learnable = learnable

        if self.learnable:
            self.mlp = nn.Linear(3, out_dim)
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
            H, W, C = x.shape

            x = x.view(-1, C)    # [H*W, C]
            y = self.mlp(x)           # [H*W, D]
            out = y.view(H, W, self.dim)
        else:
            out = [x] if self.include_xyz else []
            for freq in self.freq_bands:
                out.append(torch.sin(freq * x))
                out.append(torch.cos(freq * x))
            out = torch.cat(out, dim=-1)

        return out