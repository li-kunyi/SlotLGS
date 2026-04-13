import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.attention import PositionalEncoding


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


class MLP(nn.Module):
    def __init__(self, n_input_dims, n_output_dims, hidden_dim):
        super(MLP, self).__init__()
        self.hidden_layer1 = nn.Linear(n_input_dims, hidden_dim)
        self.hidden_layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, n_output_dims)

    def forward(self, x):
        x = F.relu(self.hidden_layer1(x))
        x = F.relu(self.hidden_layer2(x))
        x = self.output_layer(x)
        return x

class HashInstanceField(nn.Module):
    def __init__(self, output_dims=16, hidden_dim=128, 
                 hash_size=16, resolution=256):
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
                                        "n_hidden_layers": 2})
        
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

        self.pe_fn = PositionalEncoding(learnable=False, num_frequencies=5, out_dim=output_dims)
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


