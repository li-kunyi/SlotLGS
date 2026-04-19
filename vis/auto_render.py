import os
import sys
import torch
import numpy as np
import imageio
from argparse import ArgumentParser
import copy

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arguments import ModelParams, PipelineParams, OptimizationParams
from gaussian_renderer import render
from scene import Scene, GaussianModel
from scene.cameras import Camera
from utils.graphics_utils import getWorld2View2


# -----------------------------
# Look-at camera
# -----------------------------
def look_at(camera_position, target, up=np.array([0, 0, 1])):
    camera_position = np.array(camera_position)
    target = np.array(target)

    forward = target - camera_position
    forward = forward / np.linalg.norm(forward)

    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)

    true_up = np.cross(right, forward)

    c2w = np.eye(4)
    c2w[:3, 0] = right
    c2w[:3, 1] = true_up
    c2w[:3, 2] = forward
    c2w[:3, 3] = camera_position

    return c2w


# -----------------------------
# Camera builder
# -----------------------------
def build_camera(c2w, view):
    w2c = np.linalg.inv(c2w)
    view.R = w2c[:3, :3]
    view.T = w2c[:3, 3]

    view.world_view_transform = torch.tensor(getWorld2View2(view.R, view.T, np.array([0.0, 0.0, 0.0]), 1.0)).transpose(0, 1).cuda()
    view.full_proj_transform = (view.world_view_transform.unsqueeze(0).bmm(view.projection_matrix.unsqueeze(0))).squeeze(0)
    view.camera_center = view.world_view_transform.inverse()[3, :3]

    return view

# -----------------------------
# Main
# -----------------------------
def generate(dataset, opt, pipeline, ckpt_path, part_name, save_mode="mp4", device="cuda"):
    output_dir = os.path.join(dataset.model_path, "vis_rendering", part_name)
    os.makedirs(output_dir, exist_ok=True)

    mp4_path = os.path.join(output_dir, "orbit.mp4")
    png_dir = os.path.join(output_dir, "frames")
    os.makedirs(png_dir, exist_ok=True)

    with torch.no_grad():

        # -----------------------------
        # Load model
        # -----------------------------
        gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type, opt)
        scene = Scene(dataset, gaussians)
        viewpoint_stack = scene.getTrainCameras()
        view = viewpoint_stack[0]

        model_params, _ = torch.load(
            os.path.join(ckpt_path, "gaussians.pth"),
            map_location=device
        )
        gaussians.restore_feature(model_params, opt)

        scene.save(0)

        xyz = gaussians.get_xyz.detach().cpu().numpy()

        center = xyz.mean(axis=0)
        dist = np.linalg.norm(xyz - center, axis=1)
        radius = np.percentile(dist, 90) * 1.2

        print(f"[INFO] center: {center}, radius: {radius}")

        # background
        background = torch.tensor([1, 1, 1], dtype=torch.float32, device=device)
        bg = torch.rand(3, device=device) if opt.random_background else background

        # -----------------------------
        # Camera path
        # -----------------------------
        num_frames = 240
        elevation = np.deg2rad(20)
        direction = np.array([1, 0, 0])

        frames = []

        for i in range(num_frames):
            angle = 2 * np.pi / num_frames

            rot_axis = np.array([0.3, 0.7, 0.2])  # 任意轴（关键！）
            rot_axis = rot_axis / np.linalg.norm(rot_axis)

            # Rodrigues旋转公式
            direction = (
                direction * np.cos(angle)
                + np.cross(rot_axis, direction) * np.sin(angle)
                + rot_axis * np.dot(rot_axis, direction) * (1 - np.cos(angle))
            )

            cam_pos = center + direction * radius

            c2w = look_at(cam_pos, center)
            cam = copy.deepcopy(view)
            cam = build_camera(c2w, cam)

            render_pkg = render(cam, gaussians, pipeline, bg, render_instance=False)

            image = render_pkg["render"]
            image = image.clamp(0, 1)
            image = (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

            frames.append(image)

            # -----------------------------
            # save png frame
            # -----------------------------
            if save_mode in ["png", "both"]:
                frame_path = os.path.join(png_dir, f"{i:05d}.png")
                imageio.imwrite(frame_path, image)

            print(f"\rRendering {i+1}/{num_frames}", end="")

        print("\n[INFO] Rendering done")

        # -----------------------------
        # save mp4
        # -----------------------------
        if save_mode in ["mp4", "both"]:
            print("[INFO] Writing mp4...")

            imageio.mimwrite(mp4_path, frames, fps=30)

            print(f"[DONE] MP4 saved: {mp4_path}")


# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    parser = ArgumentParser("Gaussian orbit renderer")
    model = ModelParams(parser)
    opt = OptimizationParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--quiet", action="store_true") 
    parser.add_argument("--gaussian_ckpt", type=str, required=True)
    parser.add_argument("--scene_name", type=str, default=None)
    parser.add_argument("--part", type=str, default=None)
    parser.add_argument(
        "--save_mode",
        type=str,
        default="mp4",
        choices=["mp4", "png", "both"],
        help="output format"
    )

    args = parser.parse_args(sys.argv[1:])

    print("[INFO] scene:", args.scene_name)
    print("[INFO] save_mode:", args.save_mode)

    dataset = model.extract(args)
    opt = opt.extract(args)
    pipeline = pipeline.extract(args)
    
    generate(dataset, opt, pipeline, args.gaussian_ckpt, args.part, args.save_mode)