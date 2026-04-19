import os
import sys
import torch
import numpy as np
import imageio
from argparse import ArgumentParser

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arguments import ModelParams, PipelineParams, OptimizationParams
from gaussian_renderer import render
from scene.gaussian_model import GaussianModel
from scene.cameras import Camera


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
def build_camera(c2w, dataset):
    return Camera(
        c2w=c2w,
        FoVx=dataset.FoVx,
        FoVy=dataset.FoVy,
        image_width=dataset.W,
        image_height=dataset.H,
        image_name="orbit"
    )


# -----------------------------
# Main
# -----------------------------
def generate(dataset, opt, pipeline, ckpt_path, save_mode="mp4", device="cuda"):

    output_dir = os.path.join(dataset.model_path, "eval_3d")
    os.makedirs(output_dir, exist_ok=True)

    mp4_path = os.path.join(output_dir, "orbit.mp4")
    png_dir = os.path.join(output_dir, "frames")
    os.makedirs(png_dir, exist_ok=True)

    with torch.no_grad():

        # -----------------------------
        # Load model
        # -----------------------------
        gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type, opt)

        model_params, _ = torch.load(
            os.path.join(ckpt_path, "gaussians.pth"),
            map_location=device
        )
        gaussians.restore_feature(model_params, opt)

        xyz = gaussians.get_xyz.detach().cpu().numpy()

        center = np.median(xyz, axis=0)
        dist = np.linalg.norm(xyz - center, axis=1)
        radius = np.percentile(dist, 95) * 1.2

        print(f"[INFO] center: {center}, radius: {radius}")

        # background
        background = torch.tensor([1, 1, 1], dtype=torch.float32, device=device)
        bg = torch.rand(3, device=device) if opt.random_background else background

        # -----------------------------
        # Camera path
        # -----------------------------
        num_frames = 120
        elevation = np.deg2rad(20)

        frames = []

        for i in range(num_frames):

            theta = 2 * np.pi * i / num_frames

            cam_pos = np.array([
                center[0] + radius * np.cos(theta) * np.cos(elevation),
                center[1] + radius * np.sin(theta) * np.cos(elevation),
                center[2] + radius * np.sin(elevation)
            ])

            c2w = look_at(cam_pos, center)
            cam = build_camera(c2w, dataset)

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

    parser.add_argument("--gaussian_ckpt", type=str, required=True)
    parser.add_argument("--scene_name", type=str, default=None)

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

    generate(dataset, opt, pipeline, args.gaussian_ckpt, args.save_mode)