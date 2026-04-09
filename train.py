#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
from random import randint
from torch.nn import functional as F
import torchvision
from utils.loss_utils import l1_loss, l2_loss, ssim, get_cluster_centroids, cosine_similarity, similarity_loss, uniformity_loss
from utils.loss_utils import entropy_loss, contrastive_clustering_loss_fast, contrastive_clustering_loss
from utils.geometry_utils import depth_to_normal, depths_to_points
from gaussian_renderer import render
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
import numpy as np
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from model.slot_attention_mem import Attention
from utils.vis_utils import visualizer_ply, visualizer_rgb, visualizer_semantic, visualizer_slot
# try:
#     from torch.utils.tensorboard import SummaryWriter
#     TENSORBOARD_FOUND = True
# except ImportError:
#     TENSORBOARD_FOUND = False
TENSORBOARD_FOUND = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint=None, debug_from=None):

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type, opt)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    if checkpoint is not None and os.path.exists(f"{checkpoint}/gaussians.pth"):
        print("Loading existing Gaussian Model.")
        (model_params, first_iter) = torch.load(f"{checkpoint}/gaussians.pth")
        gaussians.restore_feature(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = scene.getTrainCameras().copy()
    ema_loss_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), initial=first_iter, total=opt.iterations, desc="Appearance Training")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        rand_idx = randint(0, len(viewpoint_stack) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, render_instance=False)

        image, viewspace_point_tensor, visibility_filter, radii = \
            render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        render_pkg["gt_image"] = gt_image

        ssim_value = ssim(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        if iteration > 15_000:
            if gaussians.ins_optimizer is None:
                if opt.use_mlp:
                    gaussians.set_mlp(opt.ins_feature_dim)
                gaussians.training_setup_ins(opt)

            # instance feature training
            ins_pkg = render(viewpoint_cam, gaussians, pipe, bg, render_instance=True, render_rgb=False)

            # Instance Feature Training
            instance_feature = ins_pkg["render_ins_feature"]  # [D, H, W]
            render_pkg["render_ins_feature"] = instance_feature
            instance_feature_flat = instance_feature.reshape(opt.ins_feature_dim, -1).permute(1, 0)  # [N, D]
            
            D, H, W = instance_feature.shape
            
            # Load gt instance masks from the camera
            gt_masks = viewpoint_cam.get_instance_masks(instance_mask_dir=dataset.im_path, levels=['m', 'l'])

            gt_instance_masks = torch.stack([gt_masks['m'], gt_masks['l']], dim=0)
            gt_instance_masks = F.interpolate(gt_instance_masks.unsqueeze(0).float(), 
                                         size=(H, W), mode="nearest").squeeze(0)
            instance_mask_flat = gt_instance_masks.cuda().long().flatten(1, 2) # Flatten
            
            # Compute contrastive clustering loss
            # loss += opt.lambda_ins * contrastive_clustering_loss_fast(instance_feature_flat[:, :D//2], instance_mask_flat[0], normalize=True)
            # loss += opt.lambda_ins * contrastive_clustering_loss_fast(instance_feature_flat[:, D//2:], instance_mask_flat[1], normalize=True)
            loss += opt.lambda_ins * contrastive_clustering_loss_fast(instance_feature_flat, instance_mask_flat[1])

            # valid_opacity = (gaussians.get_opacity)[visibility_filter]
            # opacity_loss = torch.exp(-(valid_opacity - 0.5) ** 2 * 10).mean()
            # loss += 0.01 * opacity_loss

        loss.backward()

        iter_end.record()

        # Gaussian Update
        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = loss.item()

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            # training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Gaussian densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter, image.shape[2], image.shape[1])

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.05, scene.cameras_extent, size_threshold, radii)
                
                if iteration % opt.opacity_reset_interval == 0:
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

                if gaussians.ins_optimizer is not None:
                    gaussians.ins_optimizer.step()
                    gaussians.ins_optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                os.makedirs(scene.model_path + "/ckpt" + str(iteration), exist_ok=True)
                torch.save((gaussians.capture_feature(), iteration), scene.model_path + "/ckpt" + str(iteration) + "/gaussians.pth")
                if gaussians.mlp is not None:
                    gaussians.save_mlp(scene.model_path + "/ckpt" + str(iteration))

            # Visualization
            if iteration % 100 == 0 and True:
                depth = render_pkg["depth"]
                depth_normal, _ = depth_to_normal(viewpoint_cam, depth, world_frame=True)
                render_pkg["depth_normals"] = depth_normal
                visualizer_rgb(render_pkg, iteration, scene.model_path)

    print("Gaussian Appearance Training Completed!")



def training_semantic(dataset, opt, pipe, checkpoint_iterations, checkpoint=None, encoder='clip'):
    first_iter = 0
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type, opt)
    scene = Scene(dataset, gaussians)

    if checkpoint is not None and os.path.exists(f"{checkpoint}/gaussians.pth"):
        print("Loading existing Gaussian Model.")
        (model_params, _) = torch.load(f"{checkpoint}/gaussians.pth")
        gaussians.restore_feature(model_params, opt)
        if opt.use_mlp:
            gaussians.set_mlp(opt.ins_feature_dim)
            gaussians.load_mlp(checkpoint)
    else:
        raise("Start Appearance Training First!")

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = scene.getTrainCameras().copy()

    # Set up Attention model
    use_ins = True
    use_rgb = opt.use_rgb
    use_geo = opt.use_geometry

    # instance feature to semantics
    Attn = Attention(feat_dim=opt.ins_feature_dim,
                     vl_feat_dim=opt.vl_feature_dim, 
                     use_geo=use_geo,
                     use_rgb=use_rgb,
                     slot_path=os.path.join(dataset.lf_path, "cluster_feats.npy")
                     ).cuda()
    
    if checkpoint is not None and os.path.exists(f"{checkpoint}/attn_module.pth"):
        print("Loading existing Attention Model.")
        Attn.load(checkpoint)

    optimizer = torch.optim.Adam(Attn.parameters(), lr=1e-3)

    total_iterations = opt.semantic_iterations
    batchsize = 8192

    # Slot Initialization
    progress_bar = tqdm(range(0, 5000), initial=0, total=5000, desc="Slot Initializing")
    first_iter += 1
    for iteration in range(5000):
        iter_start.record()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        rand_idx = randint(0, len(viewpoint_stack) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        with torch.no_grad():
            render_pkg = render(viewpoint_cam, gaussians, pipe, bg, render_instance=False)
            render_pkg["gt_image"] = viewpoint_cam.original_image.cuda()

            # instance feature training
            ins_pkg = render(viewpoint_cam, gaussians, pipe, bg, render_instance=True, render_rgb=False)
            instance_feature = ins_pkg["render_ins_feature"]  # [D, H, W]
            render_pkg["render_ins_feature"] = instance_feature

            image = render_pkg["render"].permute(1, 2, 0).cuda()

            instance_feature = render_pkg["render_ins_feature"].permute(1, 2, 0).cuda()
            
            gt_image = render_pkg["gt_image"].permute(1, 2, 0).cuda()
            H, W, C = gt_image.shape

            depth = render_pkg["depth"]
            pts_world = depths_to_points(viewpoint_cam, depth, world_frame=True)
            render_pkg["render_pts_world"] = pts_world.reshape(H, W, 3).permute(2, 0, 1)
            pts_map = pts_world.reshape(H, W, 3).cuda()
            
            # Load target Vision-Language feature map
            name = viewpoint_cam.image_name.split('.')[0]
            vl_feature, valid_mask, seg_map = Attn.load_target_feature(dataset.lf_path, name, H, W, encoder=encoder)
            render_pkg["vl_feature"] = vl_feature
            vl_feature = vl_feature.permute(1, 2, 0).cuda()

            # Sample pixels
            random_idx = torch.randint(0, H * W, [batchsize])
            valid_sample = valid_mask.reshape(-1)[random_idx]
            rgb_sample = image.reshape(-1, 3)[random_idx][valid_sample]
            pts_sample = pts_map.reshape(-1, 3)[random_idx][valid_sample]
            ins_feature_sample = instance_feature.reshape(-1, instance_feature.shape[-1])[random_idx][valid_sample]  # [H*W, D]
            vl_feature_sample = vl_feature.reshape(-1, vl_feature.shape[-1])[random_idx][valid_sample]

            # Attention forward
            if use_rgb:
                app_feature_sample = Attn.rgb_embed(rgb_sample)
                app_feature_sample = torch.cat([app_feature_sample, ins_feature_sample], dim=-1)
            else:
                app_feature_sample = ins_feature_sample
            
            if use_geo:
                geo_feature_sample = Attn.PEn(pts_sample)
                app_feature_sample = torch.cat([app_feature_sample, geo_feature_sample], dim=-1)

            error = Attn.slot_init(app_feature_sample.float(), vl_feature_sample.float())

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Error": f"{error*1e5:.{7}f}"})
                progress_bar.update(10)
                
    progress_bar.close()

    slot_optimizer = Attn.set_slots_parameters(lr=1e-5)

    # Attention Training
    progress_bar = tqdm(range(first_iter, total_iterations), initial=first_iter, total=total_iterations, desc="Semantic Training")
    first_iter += 1
    for iteration in range(first_iter, total_iterations + 1):
        iter_start.record()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        rand_idx = randint(0, len(viewpoint_stack) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        with torch.no_grad():
            render_pkg = render(viewpoint_cam, gaussians, pipe, bg, render_instance=False)
            render_pkg["gt_image"] = viewpoint_cam.original_image.cuda()

            # instance feature training
            ins_pkg = render(viewpoint_cam, gaussians, pipe, bg, render_instance=True, render_rgb=False)
            instance_feature = ins_pkg["render_ins_feature"]  # [D, H, W]
            render_pkg["render_ins_feature"] = instance_feature

            image = render_pkg["render"].permute(1, 2, 0).cuda()

            instance_feature = render_pkg["render_ins_feature"].permute(1, 2, 0).cuda()
            
            gt_image = render_pkg["gt_image"].permute(1, 2, 0).cuda()
            H, W, C = gt_image.shape

            depth = render_pkg["depth"]
            pts_world = depths_to_points(viewpoint_cam, depth, world_frame=True)
            render_pkg["render_pts_world"] = pts_world.reshape(H, W, 3).permute(2, 0, 1)
            pts_map = pts_world.reshape(H, W, 3).cuda()
            
            # Load target Vision-Language feature map
            name = viewpoint_cam.image_name.split('.')[0]
            vl_feature, valid_mask, seg_map = Attn.load_target_feature(dataset.lf_path, name, H, W, encoder=encoder)
            render_pkg["vl_feature"] = vl_feature
            vl_feature = vl_feature.permute(1, 2, 0).cuda()

            # Sample pixels
            random_idx = torch.randint(0, H * W, [batchsize])
            valid_sample = valid_mask.reshape(-1)[random_idx]
            rgb_sample = image.reshape(-1, 3)[random_idx][valid_sample]
            pts_sample = pts_map.reshape(-1, 3)[random_idx][valid_sample]
            ins_feature_sample = instance_feature.reshape(-1, instance_feature.shape[-1])[random_idx][valid_sample]  # [H*W, D]
            vl_feature_sample = vl_feature.reshape(-1, vl_feature.shape[-1])[random_idx][valid_sample]

        # Attention forward
        if use_rgb:
            app_feature_sample = Attn.rgb_embed(rgb_sample)
            app_feature_sample = torch.cat([app_feature_sample, ins_feature_sample], dim=-1)
        else:
            app_feature_sample = ins_feature_sample
        
        if use_geo:
            geo_feature_sample = Attn.PEn(pts_sample)
            app_feature_sample = torch.cat([app_feature_sample, geo_feature_sample], dim=-1)

        out_feature, attn_weights = Attn(app_feature_sample.float())

        # Vision-Language loss
        recon_vl_feature = out_feature['vl']
        vl_loss = cosine_similarity(recon_vl_feature, vl_feature_sample) + l1_loss(recon_vl_feature, vl_feature_sample)
        loss = opt.lambda_vl_recon * vl_loss

        # Slot Regularization
        # Entropy loss: each pixel only focus one slot
        ent_loss = entropy_loss(attn_weights, eps=1e-8, reduction='mean')
        loss += opt.lambda_ent * ent_loss

        # Attention loss: all slots being used
        # if iteration < 2000:
        #     attn_loss = (1 - attn_weights.max(dim=0).values).mean()
        #     loss += opt.lambda_attn * attn_loss

        loss.backward()

        optimizer.step()
        slot_optimizer.step()
        optimizer.zero_grad(set_to_none = True)
        slot_optimizer.zero_grad(set_to_none = True)

        # Slots Update
        with torch.no_grad():
            # Log and Save
            ema_loss_for_log = loss.item()

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                os.makedirs(scene.model_path + "/ckpt_attn" + str(iteration), exist_ok=True)
                Attn.save(scene.model_path + "/ckpt_attn" + str(iteration))

            # Visualization
            if iteration % 500 == 0:
                visualizer_semantic(render_pkg, iteration, scene.model_path, Attn, use_rgb=use_rgb, use_geo=use_geo, use_ins=use_ins)

            if iteration % 1000 == 0:
                visualizer_slot(render_pkg, iteration, scene.model_path, Attn, use_rgb=use_rgb, use_geo=use_geo, use_ins=use_ins)
            
            if iteration % 5000 == 0 and False:
                gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type, opt)
                if checkpoint:
                    (model_params, first_iter) = torch.load(f"{checkpoint}/gaussians.pth")
                    gaussians.restore_feature(model_params, opt)

                visualizer_ply(gaussians, iteration, save_dir, Attn, use_rgb=use_rgb, use_geo=use_geo, use_ins=use_ins)
                del gaussians
                
    print("\n[ITER {}] Saving Checkpoint".format(iteration))
    os.makedirs(scene.model_path + "/ckpt_attn" + str(iteration), exist_ok=True)
    Attn.save(scene.model_path + "/ckpt_attn" + str(iteration))

    print("Gaussian Vision-Language Training Completed!")

        
def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    # if TENSORBOARD_FOUND:
    #     tb_writer = SummaryWriter(args.model_path)
    # else:
    #     print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[15_000, 30_000])
    parser.add_argument("--ckpt_path", type=str, default = None)
    parser.add_argument("--encoder", type=str, default = 'clip')
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    dataset_args = lp.extract(args)
    opt_args = op.extract(args)
    pipe_args = pp.extract(args)

    dataset_args.im_path = os.path.join(dataset_args.im_path, args.encoder)
    dataset_args.lf_path = os.path.join(dataset_args.lf_path, args.encoder)

    opt_args.vl_feature_dim = 512 if args.encoder == 'clip' else 768

    training(dataset_args, opt_args, pipe_args, args.test_iterations, args.save_iterations, args.checkpoint_iterations, f"{args.ckpt_path}/ckpt15000", args.debug_from)

    training_semantic(dataset_args, opt_args, pipe_args, [5_000, 10_000], checkpoint=f"{args.ckpt_path}/ckpt30000", encoder=args.encoder)
