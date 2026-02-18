import time
import numpy as np
import os
import random
import torch
import torchvision
from pathlib import Path
from argparse import ArgumentParser
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import torch.nn.functional as F
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import GaussianModel and render directly from their files, not from the package
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
from gaussian_renderer import render
from scene.gaussian_model import GaussianModel
from scene import Scene
from utils.general_utils import safe_state
from model.slot_attention_mem import Attention
from utils.sh_utils import SH2RGB
from eval.lerf_ovs import evalute, get_queries, eval_gt_lerfdata, get_query_text_features
from torchvision.utils import draw_segmentation_masks


scene_gt_frames = {
    "waldo_kitchen": ["frame_00053", "frame_00066", "frame_00089", "frame_00140", "frame_00154"],
    "ramen": ["frame_00006", "frame_00024", "frame_00060", "frame_00065", "frame_00081", "frame_00119", "frame_00128"],
    "figurines": ["frame_00041", "frame_00105", "frame_00152", "frame_00195"],
    "teatime": ["frame_00002", "frame_00025", "frame_00043", "frame_00107", "frame_00129", "frame_00140"]
}
    
SCENE_TEXTS = {
    "waldo_kitchen": ['Stainless steel pots', 'dark cup', 'refrigerator', 'frog cup', 'pot', 'spatula', 'plate', \
            'spoon', 'toaster', 'ottolenghi', 'plastic ladle', 'sink', 'ketchup', 'cabinet', 'red cup', \
            'pour-over vessel', 'knife', 'yellow desk'],
    "ramen": ['nori', 'sake cup', 'kamaboko', 'corn', 'spoon', 'egg', 'onion segments', 'plate', \
            'napkin', 'bowl', 'glass of water', 'hand', 'chopsticks', 'wavy noodles'],
    "figurines": ['jake', 'pirate hat', 'pikachu', 'rubber duck with hat', 'porcelain hand', \
                'red apple', 'tesla door handle', 'waldo', 'bag', 'toy cat statue', 'miffy', \
                'green apple', 'pumpkin', 'rubics cube', 'old camera', 'rubber duck with buoy', \
                'red toy chair', 'pink ice cream', 'spatula', 'green toy chair', 'toy elephant'],
    "teatime": ['sheep', 'yellow pouf', 'stuffed bear', 'coffee mug', 'tea in a glass', 'apple', 
            'coffee', 'hooves', 'bear nose', 'dall-e brand', 'plate', 'paper napkin', 'three cookies', \
            'bag of cookies'] # 'dall-e brand' is not in the dataset
}


def seed_everything(seed_value):
    """
    Function that seed everything
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        

def cosine_similarity(pred, target, batch_size=1024):
    N, C = pred.shape
    cos_sim_list = []

    if target.dim() == 1:
        target = target.unsqueeze(0)  # [1, C]

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        pred_batch = pred[start:end]            # [batch_size, C]

        target_batch = target.expand(end - start, -1)
        cos_sim_batch = F.cosine_similarity(F.normalize(pred_batch, dim=-1), F.normalize(target_batch, dim=-1), dim=-1)
        cos_sim_list.append(cos_sim_batch)

    cos_sim_map = torch.cat(cos_sim_list, dim=0)  # [N]
    return cos_sim_map


def rendering(output_dir, views, gaussians, pipe, bg, 
              scene_name, masks=[], threshold=0.1):        
    target_text = SCENE_TEXTS[scene_name]
    color_map = get_queries(scene_name)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        view_name = (view.image_name).split('.')[0]
        view_dir = os.path.join(output_dir, view_name)
        os.makedirs(view_dir, exist_ok=True)
        
        render_pkg = render(view, gaussians, pipe, bg, render_instance=False)
        full_image = render_pkg["render"]
        gt_image = view.original_image
        torchvision.utils.save_image(full_image, os.path.join(view_dir, f"color.png"))

        for i, query in enumerate(target_text):
            mask = masks[i]

            render_pkg = render(view, gaussians, pipe, bg, render_instance=False, mask=mask)
            image = render_pkg["render"]  
            alpha = render_pkg["alpha"]
            binary_mask = (alpha > threshold)

            render_path = os.path.join(view_dir, "rgb")  # Inside frame_name folder
            os.makedirs(render_path, exist_ok=True)                      
            torchvision.utils.save_image(image, os.path.join(render_path, f"{query}.png"))

            # Save mask for IoU compute
            mask_path = os.path.join(view_dir, "mask")   # Inside frame_name folder
            os.makedirs(mask_path, exist_ok=True)
            torchvision.utils.save_image(binary_mask.to(torch.float32), os.path.join(mask_path, f"{query}.png"))

            # Visualize query heat map
            img_uint8 = (gt_image.clamp(0, 1) * 255).to(torch.uint8)

            # Just to match the same colours as opengaussian
            color = color = [color_map.get(query, (255, 255, 255))] #get_color(query, color_map)
            overlay = draw_segmentation_masks(
                img_uint8.cpu(),
                masks=binary_mask.cpu(),
                alpha=0.5,
                colors=color)  # return a tensor uint8 [3, H, W]

            # convert to float [0,1] to save
            overlay = overlay.float() / 255.0

            query_path = os.path.join(view_dir, "query") # Inside frame_name folder
            os.makedirs(query_path, exist_ok=True)
            torchvision.utils.save_image(overlay, os.path.join(query_path, f"{query}_overlay.png"))


def smooth_mask(valid_map_3d, indices, thresh=0.4, device="cuda"):
    # smooth the relevancy map, similar to in 2D
    relv_map_smoothed = torch.zeros_like(valid_map_3d)

    relv_1d = valid_map_3d
    neighbors_vals = relv_1d[indices]  
    neighbors_avg = neighbors_vals.mean(dim=1)  
    relv_map_smoothed = 0.5 * (relv_1d + neighbors_avg)

    output = relv_map_smoothed
    output = output - torch.min(output)
    output = output / (torch.max(output) + 1e-9)
    output = output * (1.0 - (-1.0)) + (-1.0)
    output = torch.clip(output, 0, 1)
    
    gs_masks_pred = (output > thresh)
    
    return gs_masks_pred


def generate(dataset, opt, pipeline, gaussian_ckpt_path, attn_ckpt_path, 
             scene_name, json_dir, text_feature_dir, render_all=False, 
             threshold=0.8, alpha_threshold=0.1, device="cuda"):    
    output_dir = os.path.join(dataset.model_path, "eval_3d")
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():        
        # load Gaussian and Attention models
        gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type, opt)
        scene = Scene(dataset, gaussians, shuffle=False)

        (model_params, first_iter) = torch.load(f"{gaussian_ckpt_path}/gaussians.pth")
        gaussians.restore_feature(model_params, opt)

        background = torch.tensor([1,1,1], dtype=torch.float32, device="cuda")
        views = scene.getTrainCameras()

        # Load Attention model
        use_ins = opt.use_instance_feature
        use_rgb = opt.use_rgb
        use_geo = opt.use_geometry

        Attn = Attention(ins_dim=opt.instance_feature_dim,
                         tgt_feat_dim=opt.target_feature_dim, 
                         num_slots=opt.slot_num, 
                         in_slot_dim=opt.instance_slot_dim, 
                         tgt_slot_dim=opt.target_slot_dim,
                         use_geo=use_geo,
                         use_rgb=use_rgb,
                         use_ins=use_ins
                         ).cuda()
        Attn.load(attn_ckpt_path)
        
        # Get per gaussian's semantic feature
        pts = gaussians.get_xyz
        instance_feature = gaussians.get_ins_feature
        shs = gaussians.get_features
        rgb = SH2RGB(shs[:, 0])

        if use_rgb:
            feature = Attn.rgb_embed(rgb.reshape(-1, 3))
            feature = torch.cat([feature, instance_feature], dim=-1)  # [H, W, C+D]
        else:
            feature = instance_feature

        if use_geo:
            geo_feature = Attn.PEn(pts)
            feature = torch.cat([feature, geo_feature], dim=-1)

        features, _ = Attn.inference(feature.reshape(-1, feature.shape[-1]).float())  # [H*W, D]
        gaussian_semantics = features['semantic']

        xyz_np = pts.cpu().numpy()
        nbrs = NearestNeighbors(n_neighbors=10).fit(xyz_np)
        _, indices = nbrs.kneighbors(xyz_np)
        indices = torch.from_numpy(indices).to(pts.device)

        color_map = get_queries(scene_name)
        gt_ann, image_shape, image_paths = eval_gt_lerfdata(Path(json_dir), Path(output_dir))  # TODO
        eval_index_list = [int(idx) for idx in list(gt_ann.keys())] # zero-based index of the image frame in the dataset (00002 -> 1)
        
        target_text, query_text_feat = get_query_text_features(scene_name, text_feature_dir)

        for j, idx in enumerate(tqdm(eval_index_list)):
            # Get current frame_name view and its gt anottations
            view = views[idx]
            # print(f"View image name: {view.image_name}, Query: frame_{idx+1:0>5}")

            view_name = (view.image_name).split('.')[0]
            view_dir = os.path.join(output_dir, view_name)
            os.makedirs(view_dir, exist_ok=True)

            render_path = os.path.join(view_dir, "rgb")  # Inside frame_name folder
            os.makedirs(render_path, exist_ok=True)  

            mask_path = os.path.join(view_dir, "mask")   # Inside frame_name folder
            os.makedirs(mask_path, exist_ok=True)

            query_path = os.path.join(view_dir, "query") # Inside frame_name folder
            os.makedirs(query_path, exist_ok=True)

            img_ann = gt_ann[f'{idx}']     # {..., 'object name': {bboxes: array, 'mask': array}, ...}
            queries = list(img_ann.keys()) # Get the object instance names

            gt_image = view.original_image

            render_pkg = render(view, gaussians, pipeline, background, render_instance=False)
            full_image = render_pkg["render"]
            torchvision.utils.save_image(full_image, os.path.join(view_dir, f"color.png"))

            # Open-Vocabulary query: mask generation
            img_ann = gt_ann[f'{idx}']     # {..., 'object name': {bboxes: array, 'mask': array}, ...}
            queries = list(img_ann.keys()) # Get the object instance names
            for query in queries:
                # Obtain the mask where the query has higher similarity
                query_feature = query_text_feat[query].to(device)

                cos_sim_map = cosine_similarity(gaussian_semantics, query_feature)             
                cos_sim_map = (cos_sim_map - cos_sim_map.min()) / (cos_sim_map.max() - cos_sim_map.min() + 1e-6)
                gaussian_mask = (cos_sim_map > threshold)

                gaussian_mask = smooth_mask(gaussian_mask.float(), indices, thresh=threshold)

                render_pkg = render(view, gaussians, pipeline, background, render_instance=False, mask=gaussian_mask)
                image = render_pkg["render"]  
                alpha = render_pkg["alpha"]
                binary_mask = (alpha > alpha_threshold)

                torchvision.utils.save_image(image, os.path.join(render_path, f"{query}.png"))
                torchvision.utils.save_image(binary_mask.to(torch.float32), os.path.join(mask_path, f"{query}.png"))

                # Visualize query heat map
                img_uint8 = (gt_image.clamp(0, 1) * 255).to(torch.uint8)

                # Just to match the same colours as opengaussian
                color = [color_map.get(query, (255, 255, 255))] #get_color(query, color_map)
                overlay = draw_segmentation_masks(
                    img_uint8.cpu(),
                    masks=binary_mask.cpu(),
                    alpha=0.5,
                    colors=color)  # return a tensor uint8 [3, H, W]
                overlay = overlay.float() / 255.0          

                torchvision.utils.save_image(overlay, os.path.join(query_path, f"{query}_overlay.png"))


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Visualization script parameters")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--json_dir", type=str, default='dataset/lerf_ovs/label')
    parser.add_argument("--mask_thresh", type=float, default=0.4)
    parser.add_argument("--scene_name", type=str, default=None)
    parser.add_argument("--encoder", type=str, default = 'clip')
    parser.add_argument("--text_feature_dir", type=str, default='eval/clip')
    parser.add_argument("--gaussian_ckpt", type=str, default='output/lerf_ovs/figurines/ckpt30000')
    parser.add_argument("--attn_ckpt", type=str, default='output/lerf_ovs/figurines/ckpt_semantic5000')
    parser.add_argument('--render_all', action='store_true', default=False)
 
    op, model, pipeline = OptimizationParams(parser), ModelParams(parser, sentinel=True), PipelineParams(parser)
    args = get_combined_args(parser)
    print("[INFO]: Evaluating file " + args.scene_name)
    # print(f"[INFO]: {args}")
    
    # Initialize system state (RNG)
    safe_state(args.quiet)
    seed_everything(seed_value=42)

    dataset_args = model.extract(args)
    opt_args = op.extract(args)
    pipe_args = pipeline.extract(args)
    
    scene_name = args.scene_name
    json_dir = os.path.join(args.json_dir, args.scene_name)
    text_feature_dir = args.text_feature_dir
    gaussian_ckpt_path = args.gaussian_ckpt
    attn_ckpt_path = args.attn_ckpt
    opt_args.target_feature_dim = 512 if args.encoder == 'clip' else 768

    generate(dataset_args, opt_args, pipe_args, gaussian_ckpt_path, attn_ckpt_path, scene_name, json_dir, text_feature_dir, render_all=args.render_all, threshold=args.mask_thresh)

    # Compute IoU, Acc
    path_gt = os.path.join(dataset_args.model_path, "eval_3d", "gt")
    eval_path = os.path.join(dataset_args.model_path, "eval_3d")
    evalute(path_gt, eval_path, args.scene_name, eval_path)
