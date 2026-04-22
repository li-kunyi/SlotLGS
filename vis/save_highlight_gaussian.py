import time
import numpy as np
import os
import random
import torch
from argparse import ArgumentParser
from sklearn.neighbors import NearestNeighbors

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arguments import ModelParams, PipelineParams, OptimizationParams
from scene.gaussian_model import GaussianModel
from eval.openclip_encoder import OpenCLIPNetwork
from scene import Scene
from model.model import Attention


SCENE_TEXTS = {
    "waldo_kitchen": ['Stainless steel pots', 'dark cup', 'refrigerator', 'frog cup'],
    "ramen": ['nori', 'sake cup', 'egg', 'bowl'],
    "figurines": ['pikachu', 'rubber duck', 'apple', 'camera'],
    "teatime": ['sheep', 'coffee mug', 'apple', 'cookies']
}


def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)


def get_mask(feature, xyz, clip_model=None, thresh=0.4, num_knn=10):
    valid_map_3d = clip_model.get_max_across_3d(feature)
    n_prompt, _ = valid_map_3d.shape

    xyz_np = xyz.cpu().numpy()
    nbrs = NearestNeighbors(n_neighbors=num_knn).fit(xyz_np)
    _, indices = nbrs.kneighbors(xyz_np)
    indices = torch.from_numpy(indices).to(valid_map_3d.device)

    relv_map_smoothed = torch.zeros_like(valid_map_3d)
    gs_masks_pred = torch.zeros_like(valid_map_3d)

    for i in range(n_prompt):
        relv_1d = valid_map_3d[i]
        neighbors_vals = relv_1d[indices]
        neighbors_avg = neighbors_vals.mean(dim=1)

        relv = 0.5 * (relv_1d + neighbors_avg)

        relv = relv - relv.min()
        relv = relv / (relv.max() + 1e-9)

        gs_masks_pred[i] = relv > thresh

    return gs_masks_pred > 0.5


def generate(dataset, opt, ckpt_path, attn_ckpt_path, scene_name,
             threshold=0.4, levels=['l'], device="cuda"):

    with torch.no_grad():
        clip_model = OpenCLIPNetwork(device)

        target_text = SCENE_TEXTS[scene_name]
        clip_model.set_positives(target_text)

        gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type, opt)
        scene = Scene(dataset, gaussians, shuffle=False)

        Attn = Attention(
            feat_dim=opt.ins_feature_dim,
            vl_feat_dim=opt.vl_feature_dim,
            num_slots=opt.slot_num,
            app_slot_dim=opt.app_slot_dim,
            vl_slot_dim=opt.vl_slot_dim,
            use_geo=opt.use_geometry,
            use_rgb=opt.use_rgb
        ).cuda()

        for level in levels:
            print(f"[INFO] Processing level: {level}")

            gaussian_ckpt_path = f"{ckpt_path}/{level}/ckpt30000"

            # ---- load gaussian ----
            (model_params, _) = torch.load(f"{gaussian_ckpt_path}/gaussians.pth")
            gaussians.restore_feature(model_params, opt)

            if opt.use_mlp:
                gaussians.set_mlp(opt.ins_feature_dim, opt.pe_type)
                gaussians.load_mlp(gaussian_ckpt_path)

            # ---- load attention ----
            Attn.load(f'{attn_ckpt_path}/{level}/ckpt_attn10000')

            pts = gaussians.get_xyz
            instance_feature = gaussians.get_ins_feature()

            feature = instance_feature

            if opt.use_geometry:
                geo_feature = Attn.PEn(pts)
                feature = torch.cat([feature, geo_feature], dim=-1)

            features, _ = Attn.inference(feature.float())
            pred_lang_feat = features['vl']

            # ---- get masks ----
            gs_mask_preds = get_mask(
                pred_lang_feat,
                pts,
                clip_model,
                thresh=threshold
            )

            for text_idx, text in enumerate(target_text):

                mask = gs_mask_preds[text_idx].bool()

                if mask.sum() == 0:
                    print(f"[WARN] Empty mask: {text}")
                    continue

                # ---- clone FULL gaussian ----
                new_gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type, opt)
                new_gaussians.active_sh_degree = gaussians.active_sh_degree

                new_gaussians._xyz = gaussians._xyz.clone()
                new_gaussians._features_dc = gaussians._features_dc.clone()
                new_gaussians._features_rest = gaussians._features_rest.clone()

                new_gaussians._scaling = gaussians._scaling.clone()
                new_gaussians._rotation = gaussians._rotation.clone()
                new_gaussians._opacity = gaussians._opacity.clone()

                # ---- COLOR MODIFICATION ----
                colors = new_gaussians._features_dc

                gray = torch.tensor([0.5, 0.5, 0.5], device=colors.device).view(1, 3)
                red = torch.tensor([1.0, 0.0, 0.0], device=colors.device).view(1, 3)

                # soft highlight
                alpha = 0.7

                colors[~mask] = colors[~mask] * 0.2 + gray * 0.8
                colors[mask] = alpha * red + (1 - alpha) * colors[mask]

                new_gaussians._features_dc = torch.clamp(colors, 0, 1)

                new_gaussians._features_rest *= 0

                # ---- save ----
                save_dir = os.path.join(gaussian_ckpt_path, "vis_highlight", text.replace(" ", "_"))
                os.makedirs(save_dir, exist_ok=True)

                torch.save(
                    (new_gaussians.capture_feature(), 30000),
                    os.path.join(save_dir, "gaussians.pth")
                )

                new_gaussians.save_ply(os.path.join(save_dir, "point_cloud.ply"))

                print(f"[SAVE] {text} | points={mask.sum().item()}")


if __name__ == "__main__":
    parser = ArgumentParser()

    model = ModelParams(parser)
    op = OptimizationParams(parser)

    parser.add_argument("--scene_name", type=str, required=True)
    parser.add_argument("--gaussian_ckpt", type=str, required=True)
    parser.add_argument("--attn_ckpt", type=str, required=True)
    parser.add_argument("--mask_thresh", type=float, default=0.5)
    parser.add_argument('--level', type=str, default='l')

    args = parser.parse_args()

    seed_everything(42)

    dataset_args = model.extract(args)
    opt_args = op.extract(args)

    levels = ['l', 'm', 's'] if args.level == 'all' else [args.level]

    generate(
        dataset_args,
        opt_args,
        args.gaussian_ckpt,
        args.attn_ckpt,
        args.scene_name,
        threshold=args.mask_thresh,
        levels=levels
    )