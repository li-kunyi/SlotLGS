import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
import imageio
from argparse import ArgumentParser

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False


# -----------------------------
# render single set
# -----------------------------
def render_set(model_path, name, iteration, views, gaussians, pipeline,
               background, train_test_exp, separate_sh,
               save_mode="png"):

    render_path = os.path.join(model_path, name, f"ours_{iteration}", "renders")
    gts_path = os.path.join(model_path, name, f"ours_{iteration}", "gt")
    video_path = os.path.join(model_path, name, f"ours_{iteration}", "render.mp4")

    if save_mode in ["png", "both"]:
        makedirs(render_path, exist_ok=True)
        makedirs(gts_path, exist_ok=True)

    frames = []

    for idx, view in enumerate(tqdm(views, desc=f"Rendering {name}")):

        rendering = render(
            view,
            gaussians,
            pipeline,
            background,
            use_trained_exp=train_test_exp,
            separate_sh=separate_sh
        )["render"]

        gt = view.original_image[0:3, :, :]

        # -----------------------------
        # PNG save
        # -----------------------------
        if save_mode in ["png", "both"]:
            torchvision.utils.save_image(
                rendering,
                os.path.join(render_path, f"{idx:05d}.png")
            )

            torchvision.utils.save_image(
                gt,
                os.path.join(gts_path, f"{idx:05d}.png")
            )

        # -----------------------------
        # video frame buffer
        # -----------------------------
        if save_mode in ["mp4", "both"]:
            img = rendering.clamp(0, 1)
            img = (img.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
            frames.append(img)

    # -----------------------------
    # save mp4
    # -----------------------------
    if save_mode in ["mp4", "both"]:
        print(f"[INFO] Writing video: {video_path}")
        imageio.mimwrite(video_path, frames, fps=30)
        print("[DONE] Video saved.")


# -----------------------------
# main render
# -----------------------------
def render_sets(dataset, iteration, pipeline,
                skip_train, skip_test, separate_sh,
                save_mode="png"):

    with torch.no_grad():

        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians,
                      load_iteration=iteration,
                      shuffle=False)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # -----------------------------
        # train views
        # -----------------------------
        if not skip_train:
            render_set(
                dataset.model_path,
                "train",
                scene.loaded_iter,
                scene.getTrainCameras(),
                gaussians,
                pipeline,
                background,
                dataset.train_test_exp,
                separate_sh,
                save_mode
            )

        # -----------------------------
        # test views
        # -----------------------------
        if not skip_test:
            render_set(
                dataset.model_path,
                "test",
                scene.loaded_iter,
                scene.getTestCameras(),
                gaussians,
                pipeline,
                background,
                dataset.train_test_exp,
                separate_sh,
                save_mode
            )


# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":

    parser = ArgumentParser("Gaussian Renderer")

    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)

    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")

    # ⭐ NEW: output mode
    parser.add_argument(
        "--save_mode",
        type=str,
        default="png",
        choices=["png", "mp4", "both"],
        help="output format"
    )

    args = get_combined_args(parser)

    print("Rendering:", args.model_path)
    print("Save mode:", args.save_mode)

    render_sets(
        model.extract(args),
        args.iteration,
        pipeline.extract(args),
        args.skip_train,
        args.skip_test,
        SPARSE_ADAM_AVAILABLE,
        args.save_mode
    )