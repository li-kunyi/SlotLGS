import clip
from src.model import ProjectionLayer
import torch, os

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load Talk2DINO projection layer
proj_name = 'vitb_mlp_infonce'
config_path = os.path.join("configs", f"{proj_name}.yaml")
weights_path = os.path.join("weights", f"{proj_name}.pth")

talk2dino = ProjectionLayer.from_config(config_path)
talk2dino.load_state_dict(torch.load(weights_path, map_location=device))
talk2dino.to(device)

# Load CLIP model
clip_model, _ = clip.load("ViT-B/16", device=device, jit=False)
tokenizer = clip.tokenize

# Example: Tokenize and project text features
texts = ["a cat"]
text_tokens = tokenizer(texts).to(device)
text_features = clip_model.encode_text(text_tokens)
projected_text_features = talk2dino.project_clip_txt(text_features)