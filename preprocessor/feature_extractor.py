import os
import random
import numpy as np
import torch
from tqdm import tqdm

class FeatureExtractor:
    def __init__(self, save_folder, model, seed=42):
        self.model = model
        self.save_folder = save_folder
        self.seed = seed

        self._seed_everything(self.seed)
        torch.set_default_dtype(torch.float32)


    def _seed_everything(self, seed_value):
        random.seed(seed_value)
        np.random.seed(seed_value)
        torch.manual_seed(seed_value)
        os.environ['PYTHONHASHSEED'] = str(seed_value)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed_value)
            torch.cuda.manual_seed_all(seed_value)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = True
    

    def create_features(self, name, method='clip', level='l'):
        try:
            sam_path = os.path.join(self.save_folder, 'SAM', name + '.npy')
            data = np.load(sam_path, allow_pickle=True).item()
            seg_images = data["seg_images"]
            seg_maps = data["seg_maps"]
            seg_map = torch.from_numpy(seg_maps[level]).to("cuda")
            tiles = torch.from_numpy(seg_images[level]).to("cuda")
            
            with torch.no_grad():
                feat = self.model.encode_image(tiles)
                img_embed = feat.detach().cpu().half()
                # feature_map, valid_mask = self.model.get_feature_map(seg_map.to("cuda"), feat.to("cuda"))

            os.makedirs(os.path.join(self.save_folder, 'features', method), exist_ok=True)
            save_path = os.path.join(self.save_folder, 'features', method, name)

            np.save(save_path + '_feats.npy', img_embed.cpu().numpy())
            np.save(save_path + '_seg_map.npy', seg_map.cpu().numpy())
            # np.save(save_path + '_feat_map.npy', {'feat_map': feature_map.cpu().numpy(),
            #                                       'valid_mask': valid_mask.cpu().numpy()})
            
        except Exception as e:
            print(f"[ WARNING ] Error embedding image {name}: {e}")
            return
            
            

    