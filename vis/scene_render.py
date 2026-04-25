import os
import sys
sys.path.append("..")

import random
import cv2
import imageio
import numpy as np
import torch
from tqdm import tqdm
from argparse import ArgumentParser
import matplotlib.pyplot as plt

from scene import Scene
from gaussian_renderer import GaussianModel, render
from arguments import ModelParams, OptimizationParams, PipelineParams
from model.model import Attention
from utils.geometry_utils import depths_to_points


# =========================
# Utils
# =========================

def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def visualize_depth(depth, dmin=0.01, dmax=20.0):
    depth = depth.squeeze().cpu().numpy()
    depth = (depth - dmin) / (dmax - dmin + 1e-8)
    depth = np.clip(depth, 0, 1)

    depth_color = plt.cm.viridis(depth)[:, :, :3]
    return (depth_color * 255).astype(np.uint8)


# =========================
# Random Projection
# =========================

def build_random_proj(in_dim, out_dim=3, seed=0):
    rng = np.random.RandomState(seed)
    W = rng.randn(in_dim, out_dim).astype(np.float32)
    W /= np.linalg.norm(W, axis=0, keepdims=True) + 1e-8
    return torch.from_numpy(W).cuda()


def apply_random_proj(W, feat, temp=0.5):
    H, W_, D = feat.shape
    feat_flat = feat.reshape(-1, D)

    proj = feat_flat @ W
    proj = torch.sigmoid(proj / temp)

    proj = proj.reshape(H, W_, 3)
    proj = (proj * 255).clamp(0, 255).byte().cpu().numpy()

    return proj


# =========================
# Evaluate
# =========================

def evaluate(dataset, opt, pipeline, ckpt_path, attn_ckpt_path,
             levels=['l'], device="cuda", output_mode="both"):

    output_root = os.path.join(dataset.model_path, "eval_vis")
    os.makedirs(output_root, exist_ok=True)

    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type, opt)
    scene = Scene(dataset, gaussians, shuffle=False)
    views = scene.getTrainCameras()

    background = torch.tensor([1, 1, 1], dtype=torch.float32, device=device)

    Attn = Attention(
        feat_dim=opt.ins_feature_dim,
        vl_feat_dim=opt.vl_feature_dim,
        num_slots=opt.slot_num,
        hidden_dim=opt.hidden_dim,
        vl_slot_dim=opt.vl_slot_dim,
        use_geo=opt.use_geometry,
        use_rgb=opt.use_rgb,
        random_init=opt.random_init,
    ).to(device)

    with torch.no_grad():
        for level in levels:
            print(f"\n[INFO] Processing level: {level}")

            save_dir = os.path.join(output_root, level)
            os.makedirs(save_dir, exist_ok=True)

            gaussian_ckpt = f"{ckpt_path}/{level}/ckpt30000"
            attn_ckpt = f"{attn_ckpt_path}/{level}/ckpt_attn10000"

            (model_params, _) = torch.load(f"{gaussian_ckpt}/gaussians.pth")
            gaussians.restore_feature(model_params, opt)

            if opt.use_mlp:
                gaussians.set_mlp(opt.ins_feature_dim, opt.pe_type)
                gaussians.load_mlp(gaussian_ckpt)

            Attn.load(attn_ckpt)

            W_ins = build_random_proj(opt.ins_feature_dim, 3, seed=0)
            W_lang = build_random_proj(opt.vl_feature_dim, 3, seed=42)

            # reset buffers per level
            rgb_frames, depth_frames, ins_frames, lang_frames = [], [], [], []

            for idx, view in enumerate(tqdm(views)):

                render_pkg = render(view, gaussians, pipeline, background)
                image = render_pkg["render"]
                depth = render_pkg["depth"]

                rgb_np = (image.permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
                depth_np = visualize_depth(depth)

                ins_pkg = render(view, gaussians, pipeline, background,
                                 render_instance=True, render_rgb=False)
                instance_feature = ins_pkg["render_ins_feature"].permute(1,2,0).cuda()

                ins_vis = apply_random_proj(W_ins, instance_feature)

                rgb = image.permute(1,2,0)
                H, W, _ = rgb.shape

                feature = instance_feature

                if opt.use_rgb:
                    rgb_embed = Attn.rgb_embed(rgb.reshape(-1, 3)).reshape(H, W, -1)
                    feature = torch.cat([rgb_embed, feature], dim=-1)

                if opt.use_geometry:
                    pts = depths_to_points(view, depth, world_frame=True)
                    geo = Attn.PEn(pts.reshape(-1, 3)).reshape(H, W, -1)
                    feature = torch.cat([feature, geo], dim=-1)

                D = feature.shape[-1]
                out, _ = Attn.inference(feature.reshape(-1, D).float())
                lang = out['vl'].reshape(H, W, -1)

                lang_vis = apply_random_proj(W_lang, lang)

                # =========================
                # SAVE CONTROL
                # =========================

                if output_mode in ["jpg", "both"]:
                    cv2.imwrite(os.path.join(save_dir, f"{idx:05d}_rgb.jpg"), rgb_np[:, :, ::-1])
                    cv2.imwrite(os.path.join(save_dir, f"{idx:05d}_depth.jpg"), depth_np[:, :, ::-1])
                    cv2.imwrite(os.path.join(save_dir, f"{idx:05d}_ins.jpg"), ins_vis[:, :, ::-1])
                    cv2.imwrite(os.path.join(save_dir, f"{idx:05d}_lang.jpg"), lang_vis[:, :, ::-1])

                if output_mode in ["mp4", "both"]:
                    rgb_frames.append(rgb_np)
                    depth_frames.append(depth_np)
                    ins_frames.append(ins_vis)
                    lang_frames.append(lang_vis)

            # =========================
            # SAVE VIDEO
            # =========================

            if output_mode in ["mp4", "both"]:
                print(f"[INFO] Saving videos for level: {level}")

                imageio.mimwrite(os.path.join(save_dir, f"{level}_rgb.mp4"), rgb_frames, fps=30)
                imageio.mimwrite(os.path.join(save_dir, f"{level}_depth.mp4"), depth_frames, fps=30)
                imageio.mimwrite(os.path.join(save_dir, f"{level}_instance.mp4"), ins_frames, fps=30)
                imageio.mimwrite(os.path.join(save_dir, f"{level}_language.mp4"), lang_frames, fps=30)

            print("[DONE] Level finished.")


# =========================
# Entry
# =========================

if __name__ == "__main__":

    parser = ArgumentParser()

    model = ModelParams(parser)
    op = OptimizationParams(parser)
    pipeline = PipelineParams(parser)

    parser.add_argument("--scene_name", type=str, required=True)
    parser.add_argument("--gaussian_ckpt", type=str, required=True)
    parser.add_argument("--attn_ckpt", type=str, required=True)
    parser.add_argument('--level', type=str, default='l')

    parser.add_argument(
        "--output_mode",
        type=str,
        default="both",
        choices=["mp4", "jpg", "both"]
    )

    args = parser.parse_args()

    dataset_args = model.extract(args)
    opt_args = op.extract(args)
    pipe_args = pipeline.extract(args)

    levels = ['l', 'm', 's'] if args.level == 'all' else [args.level]

    evaluate(
        dataset_args,
        opt_args,
        pipe_args,
        args.gaussian_ckpt,
        args.attn_ckpt,
        levels=levels,
        output_mode=args.output_mode
    )