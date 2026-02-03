import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os


class Attention(nn.Module):
    def __init__(self, in_feat_dim, tgt_feat_dim, num_slots, in_slot_dim, tgt_slot_dim, iters=3, train=True):
        super().__init__()
        self.slot_iters = iters
        self.num_slots = num_slots
        self.avg_attn_mass = torch.zeros(num_slots, 1, device='cuda:0')
        self.attn_count = torch.zeros(num_slots, 1, device='cuda:0')
        self.attn_max = torch.zeros(num_slots, 1, device='cuda:0')
        self.densify_count = torch.zeros(num_slots, 1, device='cuda:0')
        
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
            nn.Linear(64, 3)
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
        query_input = F.normalize(self.linear_in_slots(self.norm_in_slots(in_slots)))  # [N, D1]
        query_tgt = F.normalize(self.linear_tgt_slots(self.norm_tgt_slots(tgt_slots)))  # [N, D2]

        # features as keys
        key_input = F.normalize(self.linear_input(self.norm_input(inputs)))  # [M, D1]
        key_tgt = F.normalize(self.linear_tgt(self.norm_tgt(targets)))  # [M, D2] 

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
        semantics = semantics / (semantics.norm(dim=-1, keepdim=True) + 1e-9)

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
    
    def get_slots(self):
        return self.in_slots, self.tgt_slots
    
    def update_slots(self, in_slots, tgt_slots):
        self.in_slots = in_slots.detach().requires_grad_(True)
        self.tgt_slots = tgt_slots.detach().requires_grad_(True)

    def add_attn_status(self, weights, features, th=0.8):
        # for pruning
        self.avg_attn_mass += weights.mean(dim=0)
        self.attn_count += 1

        weight_max = torch.max(weights, dim=0)
        self.attn_max = torch.max(weight_max, self.attn_max)

        # for densification
        for i in range(self.num_slots):
            # mask out high response features
            w_i = weights[:, i]
            valid_mask = (w_i > th)
            coords = valid_mask.nonzero(as_tuple=False).squeeze()
            
            if coords.numel() == 0:
                continue

            selected_features = features[coords]

            # check if these features are from same instance
            selected_features = F.normalize(selected_features, dim=-1)
            similarity = torch.matmul(selected_features, selected_features.T)

            if (similarity < 0.5).any():
                self.densify_count[i] += 1
            
    def densification_and_prune(self, mass_th=0.5, max_th=0.5, densify_th=500):
        num_slot = self.in_slots.shape[0]
        avg_attn_mass = self.avg_attn_mass / (self.attn_count + 1)

        # Minimum slots: 16
        if num_slot > 16:
            # Prune
            mass_valid_mask = (avg_attn_mass > mass_th)
            max_valid_mask = (self.attn_max > max_th)
            valid_mask = torch.logical_or(mass_valid_mask, max_valid_mask)

            self.in_slots = self.in_slots[valid_mask]
            self.tgt_slots = self.tgt_slots[valid_mask]

            self.densify_count = self.densify_count[valid_mask]

        # Maximum slots: 128
        num_slot = self.in_slots.shape[0]
        if num_slot >= 128:
            _, top_indices = torch.topk(avg_attn_mass, k=128, largest=True)
            
            self.in_slots = self.in_slots[top_indices]
            self.tgt_slots = self.tgt_slots[top_indices]

        elif num_slot < 128:
            # Densify
            densify_mask = (self.densify_count > densify_th)
            if densify_mask.any():
                new_in_slots = self.in_slots[densify_mask]
                new_tgt_slots = self.tgt_slots[densify_mask]

                new_num, in_slot_dim = new_in_slots.shape
                new_num, tgt_slot_dim = new_tgt_slots.shape

                new_in_slots = torch.tanh(new_in_slots + torch.randn(new_num, in_slot_dim, requires_grad=True, device='cuda:0'))
                new_tgt_slots = torch.tanh(new_tgt_slots + torch.randn(new_num, tgt_slot_dim, requires_grad=True, device='cuda:0'))

                self.in_slots[densify_mask] = torch.tanh(new_in_slots + torch.randn(new_num, in_slot_dim, requires_grad=True, device='cuda:0'))
                self.tgt_slots[densify_mask] = torch.tanh(new_tgt_slots + torch.randn(new_num, tgt_slot_dim, requires_grad=True, device='cuda:0'))

                self.in_slots = torch.cat([self.in_slots, new_in_slots], dim=0)
                self.tgt_slots = torch.cat([self.tgt_slots, new_tgt_slots], dim=0)

        # Reset status
        self.num_slots = self.in_slots.shape[0]

        self.avg_attn_mass = torch.zeros(self.num_slots, 1, device='cuda:0')
        self.attn_count = torch.zeros(self.num_slots, 1, device='cuda:0')
        self.attn_max = torch.zeros(self.num_slots, 1, device='cuda:0')
        self.densify_count = torch.zeros(self.num_slots, 1, device='cuda:0')

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
