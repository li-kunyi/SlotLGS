import os
import sys
import torch
import random
import torchvision
import numpy as np
from tqdm import tqdm
from PIL import Image
from pathlib import Path
import torch.nn.functional as F
import torchvision.transforms as T
from argparse import ArgumentParser
from sklearn.decomposition import PCA
from torchvision.utils import draw_segmentation_masks
from scene import Scene
from utils.general_utils import safe_state
from eval.lerf_ovs import get_query_text_features, get_queries, eval_gt_lerfdata, evalute
from gaussian_renderer import GaussianModel, render
from arguments import ModelParams, OptimizationParams, PipelineParams, get_combined_args
from model.slot_attention_mem import Attention


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

def cosine_similarity(pred, target):
    """
    Computes a per-pixel cosine similarity map between a predicted feature map and a single target feature vector.
        :param pred: Predicted image of shape [H, W, 512]
        :param target: Target iamge of shape [1, 512]
    
    return: Cosinus similarity matrix of shaoe [H, W, 1]
    [INFO]: Range []
    """
    H, W, _ = pred.shape
    target_expanded = target.expand(H, W, -1)  # [H, W, C]
    cos_sim_map = F.cosine_similarity(pred, target_expanded, dim=-1)  # [H, W]
    return cos_sim_map

def get_color(query, color_map): 
    if query == 'stuffed bear': 
        color = "red"
    elif query == 'coffee mug':  # fixed typo
        color = 'pink'
    elif query == 'chopsticks': 
        color = 'green'
    elif query == 'nori': 
        color = 'green'
    elif query == 'rubber duck': 
        color = 'yellow'
    elif query == 'red toy chair': 
        color = (251, 255, 0)
    else:
        color = [color_map.get(query, (255, 255, 255))]
    return color


def generate(dataset, opt, pipeline, checkpoint, checkpoint_semantic, scene_name, json_dir, text_feature_dir, threshold=0.8, device="cuda"):    
    output_dir = os.path.join(dataset.model_path, "eval")
    os.makedirs(output_dir, exist_ok=True)

    pca = PCA(n_components=3)
    with torch.no_grad():
        # Load Gaussian model
        gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type, opt)
        scene = Scene(dataset, gaussians, shuffle=False)

        (model_params, first_iter) = torch.load(f"{checkpoint}/gaussians.pth")
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
        
        if checkpoint_semantic and os.path.exists(f"{checkpoint_semantic}/attn_module.pth"):
            Attn.load(checkpoint_semantic)

        color_map = get_queries(scene_name)
        gt_ann, image_shape, image_paths = eval_gt_lerfdata(Path(json_dir), Path(output_dir))  # TODO
        eval_index_list = [int(idx) for idx in list(gt_ann.keys())] # zero-based index of the image frame in the dataset (00002 -> 1)
        
        target_text, query_text_feat = get_query_text_features(scene_name, text_feature_dir)

        for j, idx in enumerate(tqdm(eval_index_list)):
            frame_name = os.path.join(output_dir, f'frame_{idx+1:0>5}') # Create folder with name: image_name
            os.makedirs(frame_name, exist_ok=True)

            render_path = os.path.join(frame_name, "rgb")  # Inside frame_name folder
            os.makedirs(render_path, exist_ok=True)

            mask_path = os.path.join(frame_name, "mask")   # Inside frame_name folder
            os.makedirs(mask_path, exist_ok=True)
            
            query_path = os.path.join(frame_name, "query") # Inside frame_name folder
            os.makedirs(query_path, exist_ok=True)
            
            # Get current frame_name view and its gt anottations
            view = views[idx]
            print(f"View image name: {view.image_name}, Query: frame_{idx+1:0>5}")

            img_ann = gt_ann[f'{idx}']
            
            # RGB rendering
            render_pkg = render(view, gaussians, pipeline, background, render_instance=False)
            image = render_pkg["render"]
            gt_image = view.original_image.cuda()

            # Save RGB
            torchvision.utils.save_image(image, os.path.join(frame_name, f"color.png"))
            
            # Feature Rendering and Attention
            ins_pkg = render(view, gaussians, pipeline, background, render_instance=True, render_rgb=False)
            instance_feature = ins_pkg["render_ins_feature"].cuda()
            instance_feature = instance_feature.permute(1, 2, 0)

            rgb = image.permute(1, 2, 0)
            H, W, _ = rgb.shape
            if use_rgb:
                feature = Attn.rgb_embed(rgb.reshape(-1, 3)).reshape(H, W, -1)
                feature = torch.cat([feature, instance_feature], dim=-1)
            else:
                feature = instance_feature
            
            if use_geo:
                pts = render_pkg["render_pts_world"].permute(1, 2, 0).cuda()
                geo_feature = Attn.PEn(pts.reshape(-1, 3)).reshape(H, W, -1)
                feature = torch.cat([feature, geo_feature], dim=-1)
            
            H, W, D = feature.shape

            out, _ = Attn.inference(feature.reshape(-1, D).float())
            pred_lang_feat_flat = out['semantic']
            pred_lang_feat = pred_lang_feat_flat.reshape(H, W, -1)

            # Visualize language feature map
            x_pca = pca.fit_transform(pred_lang_feat_flat.cpu().numpy())
            feat_vis = torch.from_numpy(x_pca).reshape(H, W, 3).permute(2, 0, 1)
            feat_vis = (feat_vis - feat_vis.min()) / (feat_vis.max() - feat_vis.min())
            torchvision.utils.save_image(feat_vis, os.path.join(frame_name, f"semantic_feature_map.png"))

            # Open-Vocabulary query: mask generation
            img_ann = gt_ann[f'{idx}']     # {..., 'object name': {bboxes: array, 'mask': array}, ...}
            queries = list(img_ann.keys()) # Get the object instance names
            for query in queries:
                # Obtain the mask where the query has higher similarity
                query_feature = query_text_feat[query].to(device)                      # Shape query: [1, 512] 

                cos_sim_map = cosine_similarity(pred_lang_feat, query_feature)             
                cos_sim_map = (cos_sim_map - cos_sim_map.min()) / (cos_sim_map.max() - cos_sim_map.min() + 1e-6)
                binary_mask = cos_sim_map > threshold  # Values: [True, False] TODO check this threshold

                # Save mask for IoU compute
                torchvision.utils.save_image(binary_mask.to(torch.float32), os.path.join(mask_path, f"{query}.png"))

                # Visualize query heat map
                target_str = f"frame_{idx+1:0>5}"
                img_path = next(path for path in image_paths if target_str in path)
                img = Image.open(img_path).convert("RGB")
                img_uint8 = T.PILToTensor()(img)

                # Just to match the same colours as opengaussian
                color = color = [color_map.get(query, (255, 255, 255))] #get_color(query, color_map)
                overlay = draw_segmentation_masks(
                    img_uint8.cpu(),
                    masks=binary_mask.cpu(),
                    alpha=0.5,
                    colors=color)  # return a tensor uint8 [3, H, W]

                # convert to float [0,1] to save
                overlay = overlay.float() / 255.0
                torchvision.utils.save_image(overlay, os.path.join(query_path, f"{query}_overlay.png"))

            
if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Visualization script parameters")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--json_dir", type=str, default='dataset/lerf_ovs/label')
    parser.add_argument("--mask_thresh", type=float, default=0.4)
    parser.add_argument("--scene_name", type=str, default=None)
    parser.add_argument("--text_feature_dir", type=str, default='eval/clip')  ##TODO
 
    op, model, pipeline = OptimizationParams(parser), ModelParams(parser, sentinel=True), PipelineParams(parser)
    args = get_combined_args(parser)
    print("[INFO]: Evaluating file " + args.scene_name)
    print(f"[INFO]: {args}")
    
    # Initialize system state (RNG)
    safe_state(args.quiet)
    seed_everything(seed_value=42)

    dataset_args = model.extract(args)
    opt_args = op.extract(args)
    pipe_args = pipeline.extract(args)
    
    scene_name = args.scene_name
    json_dir = os.path.join(args.json_dir, args.scene_name)
    text_feature_dir = args.text_feature_dir
    ckpt_path = f"{dataset_args.model_path}/ckpt30000"
    ckpt_semantic_path = f"{dataset_args.model_path}/ckpt_semantic1500"

    # Generate Mask for each queries
    generate(dataset_args, opt_args, pipe_args, ckpt_path, ckpt_semantic_path, scene_name, json_dir, text_feature_dir, threshold=args.mask_thresh)

    # Compute IoU, Acc
    path_gt = os.path.join(dataset_args.model_path, "eval", "gt")
    eval_path = os.path.join(dataset_args.model_path, "eval")
    evalute(path_gt, eval_path, args.scene_name, eval_path)