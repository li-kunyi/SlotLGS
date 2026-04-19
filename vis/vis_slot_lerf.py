import json
import os
import sys
sys.path.append("..")
import glob
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Union
from argparse import ArgumentParser
import cv2
import numpy as np
import torch
from torch.nn import functional as F
import torchvision
import matplotlib.pyplot as plt
from tqdm import tqdm
import eval.colormaps as colormaps
from eval.openclip_encoder import OpenCLIPNetwork
from eval.utils import smooth, colormap_saving, vis_mask_save, polygon_to_mask, stack_mask, show_result
from scene import Scene
from gaussian_renderer import GaussianModel, render
from arguments import ModelParams, OptimizationParams, PipelineParams
from model.model import Attention
from utils.geometry_utils import depths_to_points
from utils.vis_utils import visualizer_ply, visualizer_rgb, visualizer_semantic, visualizer_slot



def eval_gt_lerfdata(json_folder: Union[str, Path] = None, ouput_path: Path = None) -> Dict:
    """
    organise lerf's gt annotations
    gt format:
        file name: frame_xxxxx.json
        file content: labelme format
    return:
        gt_ann: dict()
            keys: str(int(idx))
            values: dict()
                keys: str(label)
                values: dict() which contain 'bboxes' and 'mask'
    """
    gt_json_paths = sorted(glob.glob(os.path.join(str(json_folder), 'frame_*.json')))
    img_paths = sorted(glob.glob(os.path.join(str(json_folder), 'frame_*.jpg')))
    gt_ann = {}
    for js_path in gt_json_paths:
        img_ann = defaultdict(dict)
        with open(js_path, 'r') as f:
            gt_data = json.load(f)
        
        h, w = gt_data['info']['height'], gt_data['info']['width']
        idx = int(gt_data['info']['name'].split('_')[-1].split('.jpg')[0]) - 1 
        for prompt_data in gt_data["objects"]:
            label = prompt_data['category']
            box = np.asarray(prompt_data['bbox']).reshape(-1)           # x1y1x2y2
            mask = polygon_to_mask((h, w), prompt_data['segmentation'])
            if img_ann[label].get('mask', None) is not None:
                mask = stack_mask(img_ann[label]['mask'], mask)
                img_ann[label]['bboxes'] = np.concatenate(
                    [img_ann[label]['bboxes'].reshape(-1, 4), box.reshape(-1, 4)], axis=0)
            else:
                img_ann[label]['bboxes'] = box
            img_ann[label]['mask'] = mask
            
            # # save for visulsization
            save_path = ouput_path / 'gt' / gt_data['info']['name'].split('.jpg')[0] / f'{label}.jpg'
            save_path.parent.mkdir(exist_ok=True, parents=True)
            vis_mask_save(mask, save_path)
        gt_ann[f'{idx}'] = img_ann

    return gt_ann, (h, w), img_paths


def compute_iou(sem_map, 
                    image, 
                    clip_model, 
                    image_name: Path = None,
                    img_ann: Dict = None, 
                    thresh : float = 0.5, 
                    colormap_options = None):
    valid_map = clip_model.get_max_across(sem_map)                 # 3xkx832x1264
    n_head, n_prompt, h, w = valid_map.shape

    # positive prompts
    chosen_iou_list, chosen_lvl_list = [], []
    for k in range(n_prompt):
        iou_lvl = np.zeros(n_head)
        mask_lvl = np.zeros((n_head, h, w))
        for i in range(n_head):
            # scale = 30
            # kernel = np.ones((scale,scale)) / (scale**2)
            # np_relev = valid_map[i][k].cpu().numpy()
            # avg_filtered = cv2.filter2D(np_relev, -1, kernel)
            # avg_filtered = torch.from_numpy(avg_filtered).to(valid_map.device)
            # valid_map[i][k] = 0.5 * (avg_filtered + valid_map[i][k])
            
            output_path_relev = image_name / 'heatmap' / f'{clip_model.positives[k]}_{i}'
            output_path_relev.parent.mkdir(exist_ok=True, parents=True)
            colormap_saving(valid_map[i][k].unsqueeze(-1), colormap_options,
                            output_path_relev)
            
            # Following LERF convention, values below 0.5 are considered background
            p_i = torch.clip(valid_map[i][k] - 0.5, 0, 1).unsqueeze(-1)
            valid_composited = colormaps.apply_colormap(p_i / (p_i.max() + 1e-6), colormaps.ColormapOptions("turbo"))
            mask = (valid_map[i][k] < 0.5).squeeze()
            valid_composited[mask, :] = image[mask, :] * 0.3  # change here to mask out background
            output_path_compo = image_name / 'composited' / f'{clip_model.positives[k]}_{i}'
            output_path_compo.parent.mkdir(exist_ok=True, parents=True)
            colormap_saving(valid_composited, colormap_options, output_path_compo)
            
            # truncate the heatmap into mask
            output = valid_map[i][k]
            # output = output - torch.min(output)
            # output = output / (torch.max(output) + 1e-9)
            # output = output * (1.0 - (-1.0)) + (-1.0)
            # output = torch.clip(output, 0, 1)

            mask_pred = (output.cpu().numpy() > thresh).astype(np.uint8)
            mask_pred = smooth(mask_pred)
            mask_lvl[i] = mask_pred
            mask_gt = img_ann[clip_model.positives[k]]['mask'].astype(np.uint8)
            
            # calculate iou
            intersection = np.sum(np.logical_and(mask_gt, mask_pred))
            union = np.sum(np.logical_or(mask_gt, mask_pred))
            iou = np.sum(intersection) / np.sum(union)
            iou_lvl[i] = iou

        score_lvl = torch.zeros((n_head,), device=valid_map.device)
        for i in range(n_head):
            score = valid_map[i, k].max()
            score_lvl[i] = score
        chosen_lvl = torch.argmax(score_lvl)
        
        chosen_iou_list.append(iou_lvl[chosen_lvl])
        chosen_lvl_list.append(chosen_lvl.cpu().numpy())
        
        # save for visulsization
        save_path = image_name / f'chosen_{clip_model.positives[k]}.png'
        vis_mask_save(mask_lvl[chosen_lvl], save_path)

    return chosen_iou_list, chosen_lvl_list


def compute_localization(sem_map, image, clip_model, image_name, img_ann):
    output_path_loca = image_name / 'localization'
    output_path_loca.mkdir(exist_ok=True, parents=True)

    valid_map = clip_model.get_max_across(sem_map)                 # 3xkx832x1264
    n_head, n_prompt, h, w = valid_map.shape
    
    # positive prompts
    acc_num = 0
    positives = list(img_ann.keys())
    for k in range(len(positives)):
        select_output = valid_map[:, k]
        
        # Find the maximum value point in the activation map after filtering
        # scale = 30
        # kernel = np.ones((scale,scale)) / (scale**2)
        np_relev = select_output.cpu().numpy()
        # avg_filtered = cv2.filter2D(np_relev.transpose(1,2,0), -1, kernel)
        avg_filtered = np_relev.transpose(1,2,0)
        
        score_lvl = np.zeros((n_head,))
        coord_lvl = []
        for i in range(n_head):
            score = avg_filtered[..., i].max()
            coord = np.nonzero(avg_filtered[..., i] == score)
            score_lvl[i] = score
            coord_lvl.append(np.asarray(coord).transpose(1,0)[..., ::-1])

        selec_head = np.argmax(score_lvl)
        coord_final = coord_lvl[selec_head]
        
        for box in img_ann[positives[k]]['bboxes'].reshape(-1, 4):
            flag = 0
            x1, y1, x2, y2 = box
            x_min, x_max = min(x1, x2), max(x1, x2)
            y_min, y_max = min(y1, y2), max(y1, y2)
            for cord_list in coord_final:
                if (cord_list[0] >= x_min and cord_list[0] <= x_max and 
                    cord_list[1] >= y_min and cord_list[1] <= y_max):
                    acc_num += 1
                    flag = 1
                    break
            if flag != 0:
                break
        
        avg_filtered = torch.from_numpy(avg_filtered[..., selec_head]).unsqueeze(-1).to(select_output.device)
        torch_relev = 0.5 * (avg_filtered + select_output[selec_head].unsqueeze(-1))
        p_i = torch.clip(torch_relev - 0.5, 0, 1)
        valid_composited = colormaps.apply_colormap(p_i / (p_i.max() + 1e-6), colormaps.ColormapOptions("turbo"))
        mask = (torch_relev < 0.5).squeeze()
        valid_composited[mask, :] = image[mask, :] * 0.3
        
        save_path = output_path_loca / f"{positives[k]}.png"
        show_result(valid_composited.cpu().numpy(), coord_final,
                    img_ann[positives[k]]['bboxes'], save_path)
    return acc_num


def vis(dataset, opt, pipeline, ckpt_path, attn_ckpt_path, scene_name, json_dir, 
             mask_thresh=0.8, levels=['l', 'm', 's'], device="cuda"):
    level = "l"
    output_dir = os.path.join(dataset.model_path, "vis_slot")
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    colormap_options = colormaps.ColormapOptions(
        colormap="turbo",
        normalize=True,
        colormap_min=-1.0,
        colormap_max=1.0,
    )

    gt_ann, image_shape, image_paths = eval_gt_lerfdata(Path(json_dir), Path(output_dir))
    eval_index_list = [int(idx) for idx in list(gt_ann.keys())]

    # openclip
    clip_model = OpenCLIPNetwork(device)

    # query text
    text_list = ['green', 'red', 'apple', 'toy chair', 'green apple', 'red apple', 'green toy chair', 'red toy chair']
    queries = clip_model.encode_text(text_list, device)
    queries = F.normalize(queries, dim=-1)

    # Gaussian
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type, opt)
    scene = Scene(dataset, gaussians, shuffle=False)
    background = torch.tensor([1,1,1], dtype=torch.float32, device="cuda")
    views = scene.getTrainCameras()
    
    # Attention
    use_rgb = opt.use_rgb
    use_geo = opt.use_geometry
    Attn = Attention(feat_dim=opt.ins_feature_dim,
                vl_feat_dim=opt.vl_feature_dim, 
                num_slots=opt.slot_num, 
                app_slot_dim=opt.app_slot_dim, 
                vl_slot_dim=opt.vl_slot_dim,
                use_geo=use_geo,
                use_rgb=use_rgb
                ).cuda()   

    with torch.no_grad():
        for j, idx in enumerate(tqdm(eval_index_list)):
            frame_name = os.path.join(output_dir, f'frame_{idx:0>5}') # Create folder with name: image_name
            os.makedirs(frame_name, exist_ok=True)

            gaussian_ckpt_path = f"{ckpt_path}/{level}/ckpt30000"
            # Load Gaussian model
            (model_params, first_iter) = torch.load(f"{gaussian_ckpt_path}/gaussians.pth")
            gaussians.restore_feature(model_params, opt)
            if opt.use_mlp:
                gaussians.set_mlp(opt.ins_feature_dim, opt.pe_type)
                gaussians.load_mlp(gaussian_ckpt_path)

            # Load Attention model                
            Attn.load(f'{attn_ckpt_path}/{level}/ckpt_attn10000')

            # Get current frame_name view and its gt anottations
            view = views[idx]
            
            # RGB rendering
            render_pkg = render(view, gaussians, pipeline, background, render_instance=False)
            image = render_pkg["render"]
            depth = render_pkg["depth"]
            pts_world = depths_to_points(view, depth, world_frame=True)
            render_pkg["render_pts_world"] = pts_world.reshape(image.shape[1], image.shape[2], 3)
            gt_image = view.original_image.cuda()
            render_pkg["gt_image"] = gt_image

            ins_pkg = render(view, gaussians, pipeline, background, render_instance=True, render_rgb=False)
            instance_feature = ins_pkg["render_ins_feature"].cuda()
            render_pkg["render_ins_feature"] = instance_feature

            import ipdb;ipdb.set_trace()
            # Slot heatmap visualization
            visualizer_slot(render_pkg, 0, frame_name, Attn, use_rgb=use_rgb, use_geo=use_geo)

            # Slot-Query heatmap
            slots = Attn.get_slots()
            slots = F.normalize(slots, dim=-1)
            similarities = torch.mm(queries.double(), slots.double().T)  # [N, M]
            # similarities = F.softmax(similarities, dim=0)

            sim = similarities.detach().cpu().numpy()
            
            # ===== threshold filtering =====
            th = 0.2  # change here
            mask = sim > th

            valid_queries = np.where(mask.any(axis=1))[0]
            valid_slots = np.where(mask.any(axis=0))[0]

            if len(valid_queries) == 0 or len(valid_slots) == 0:
                print("No similarity above threshold")
                return

            sim_filtered = sim[np.ix_(valid_queries, valid_slots)]

            # ===== plotting =====
            plt.figure(figsize=(max(6, len(valid_slots)*0.6), 
                                max(4, len(valid_queries)*0.6)))

            plt.imshow(sim_filtered, cmap="Reds", aspect="auto", vmin=0.0, vmax=1.0)

            cbar = plt.colorbar()
            cbar.set_label("Similarity", rotation=270, labelpad=15)

            plt.xticks(
                np.arange(len(valid_slots)),
                [f"Slot {i}" for i in valid_slots],
                rotation=90
            )

            plt.yticks(
                np.arange(len(valid_queries)),
                [f"Query \"{i}\" " for i in text_list]
            )

            # annotate only non-zero entries
            for i in range(sim_filtered.shape[0]):
                for j in range(sim_filtered.shape[1]):
                    val = sim_filtered[i, j]
                    # if val > th:
                    plt.text(j, i, f"{val:.2f}",
                            ha="center", va="center",
                            color="black", fontsize=6)

            plt.title("Sparse Slot-Query Similarity (Filtered)")
            plt.tight_layout()
            plt.savefig(f"{frame_name}/log_images/slot_heatmap_sparse.png", dpi=300)
            plt.close()


def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


if __name__ == "__main__":    
    parser = ArgumentParser(description="prompt any label")
    model = ModelParams(parser)
    op = OptimizationParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--json_dir", type=str, default='dataset/lerf_ovs/label')
    parser.add_argument("--mask_thresh", type=float, default=0.6)
    parser.add_argument("--scene_name", type=str, default=None)
    parser.add_argument("--encoder", type=str, default = 'clip')
    parser.add_argument("--gaussian_ckpt", type=str, default='output/lerf_ovs/figurines/ckpt30000')
    parser.add_argument("--attn_ckpt", type=str, default='output/lerf_ovs/figurines/ckpt_attn5000')
    parser.add_argument('--level', type=str, default='l')

    args = parser.parse_args(sys.argv[1:])
    print("[INFO]: Evaluating file " + args.scene_name)

    # Initialize system state (RNG)
    # safe_state(args.quiet)
    seed_everything(seed_value=42)

    dataset_args = model.extract(args)
    opt_args = op.extract(args)
    pipe_args = pipeline.extract(args)
    
    scene_name = args.scene_name
    json_dir = os.path.join(args.json_dir, args.scene_name)
    gaussian_ckpt_path = args.gaussian_ckpt
    attn_ckpt_path = args.attn_ckpt
    opt_args.target_feature_dim = 512 if args.encoder == 'clip' else 768
    levels = ['l', 'm', 's'] if args.level == 'all' else [args.level]

    vis(dataset_args, opt_args, pipe_args, gaussian_ckpt_path, attn_ckpt_path, scene_name, json_dir, 
             levels=levels, mask_thresh=args.mask_thresh)
