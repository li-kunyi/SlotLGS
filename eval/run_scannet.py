import os
import torch
import random
import json
import numpy as np
import torch.nn.functional as F
from argparse import ArgumentParser
from sklearn.decomposition import PCA
from utils.general_utils import safe_state
from gaussian_renderer import GaussianModel, render
from arguments import ModelParams, OptimizationParams, PipelineParams, get_combined_args
from model.slot_attention_mem import Attention
from evaluator3d import GaussianEvaluationProtocol
from utils.sh_utils import SH2RGB
from plyfile import PlyData, PlyElement


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

nyu40_dict = {
    0: "unlabeled", 1: "wall", 2: "floor", 3: "cabinet", 4: "bed", 5: "chair",
    6: "sofa", 7: "table", 8: "door", 9: "window", 10: "bookshelf",
    11: "picture", 12: "counter", 13: "blinds", 14: "desk", 15: "shelves",
    16: "curtain", 17: "dresser", 18: "pillow", 19: "mirror", 20: "floormat",
    21: "clothes", 22: "ceiling", 23: "books", 24: "refrigerator", 25: "television",
    26: "paper", 27: "towel", 28: "showercurtain", 29: "box", 30: "whiteboard",
    31: "person", 32: "nightstand", 33: "toilet", 34: "sink", 35: "lamp",
    36: "bathtub", 37: "bag", 38: "otherstructure", 39: "otherfurniture", 40: "otherprop"
}

# ScanNet 20 classes
scannet19_dict = {
    1: "wall", 2: "floor", 3: "cabinet", 4: "bed", 5: "chair",
    6: "sofa", 7: "table", 8: "door", 9: "window", 10: "bookshelf",
    11: "picture", 12: "counter", 14: "desk", 16: "curtain",
    24: "refrigerator", 28: "shower curtain", 33: "toilet", 34: "sink",
    36: "bathtub", # 39: "otherfurniture"
}


# Auxiliary function to load the model
def to_numpy(x):
    """Convert input to NumPy array: handles torch Tensors and sequences."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def sigmoid(x):  
    return 1 / (1 + np.exp(-x))  

def write_ply(vertex_data, output_path):
    vertices = []
    for vertex in vertex_data:
        r = (vertex['ins_feat_r'] + 1)/2 * 255
        g = (vertex['ins_feat_g'] + 1)/2 * 255
        b = (vertex['ins_feat_b'] + 1)/2 * 255
        new_vertex = (vertex['x'], vertex['y'], vertex['z'], r, g, b)
        vertices.append(new_vertex)
    
    vertex_dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    new_vertex_data = np.array(vertices, dtype=vertex_dtype)
    
    el = PlyElement.describe(new_vertex_data, 'vertex')
    PlyData([el], text=True).write(output_path)

def read_labels_from_ply(file_path):
    ply_data = PlyData.read(file_path)
    vertex_data = ply_data['vertex'].data
    # Extract the coordinates and labels of the points. The labels are from 1 to 40 for the NYU40 dataset, with 0 being invalid.
    points = np.vstack([vertex_data['x'], vertex_data['y'], vertex_data['z']]).T
    labels = vertex_data['label']
    return points, labels

def load_scannet_gt(gt_file_path, target_id=19):
    # (1) GT ply
    points, labels = read_labels_from_ply(gt_file_path)
    # (2) note: 19 & 15 & 10 classes
    # Given the category ID that needs to be queried (relative to the original NYU40), obtain the corresponding category name.
    if target_id == 10: 
        target_id = [1, 2, 4, 5, 6, 7, 8, 9, 10, 33]                                    # 10
    elif target_id == 15:
        target_id = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 33, 34]                 # 15
    else: 
        target_id = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36] # 19

    target_dict = {key: nyu40_dict[key] for key in target_id} # dict{text_id: target_names} ≈ p.e. {..., 1: "wall", ...}
    target_names = list(target_dict.values())                 # [..., target_name, ...]

    # (3) Update gt label: 1-19 nyu40_dict labels, 0 unlabelled data
    # Obtained new point cloud labels, taking 19 categories as an example, where updated_labels are labels 0, 1-19.
    target_id_mapping = {value: index + 1 for index, value in enumerate(target_id)}
    updated_labels = np.zeros_like(labels)
    for original_value, new_value in target_id_mapping.items():
        updated_labels[labels == original_value] = new_value
    updated_gt_labels = torch.from_numpy(updated_labels.astype(np.int64)).cuda()

    unique, counts = torch.unique(updated_gt_labels.cpu(), return_counts=True) # where 0 unlabeled
    print("GT label distribution:", dict(zip(unique.tolist(), counts.tolist())))

    return updated_gt_labels, target_names, points

def load_query_text_features(target_names, json_dir):
    """
    target_names =     ['wall', ...]
    query_text_feats = [(...),  ...]
    """
    with open(json_dir, 'r') as f:
        data_loaded = json.load(f)
    all_texts = list(data_loaded.keys())                                                      # [num_text]
    text_features = torch.from_numpy(np.array(list(data_loaded.values()))).to(torch.float32)  # [num_text, 512]
    
    query_text_feats = []
    for i, text in enumerate(target_names):
        feat = text_features[all_texts.index(text)].unsqueeze(0)
        query_text_feats.append(feat)

    query_text_feats = torch.stack(query_text_feats, dim=0).cuda()
    return query_text_feats

def evaluate(dataset, opt, checkpoint, gt_file_path, text_feature_dir):    
    output_dir = os.path.join(dataset.model_path, "eval")
    os.makedirs(output_dir, exist_ok=True)

    pca = PCA(n_components=3)
    with torch.no_grad():
        evaluator = GaussianEvaluationProtocol()

        # Load Gaussian model
        gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type, opt)
        (model_params, first_iter) = torch.load(f"{checkpoint}/gaussians.pth")
        gaussians.restore_feature(model_params, opt)

        # Load Attention model
        use_ins = opt.use_instance_feature
        use_rgb = opt.use_rgb
        use_geo = opt.use_geometry
        if use_ins:
            in_feat_dim = opt.instance_feature_dim
            if use_rgb:
                in_feat_dim += 3
        else:
            in_feat_dim = 3

        Attn = Attention(in_feat_dim=in_feat_dim,
                         tgt_feat_dim=opt.target_feature_dim, 
                         num_slots=opt.slot_num, 
                         in_slot_dim=opt.instance_slot_dim, 
                         tgt_slot_dim=opt.target_slot_dim,
                         use_geo=use_geo
                         ).cuda()
        if checkpoint and os.path.exists(f"{checkpoint}/attn_module.pth"):
            Attn.load(checkpoint)

 
        instance_feature = gaussians.get_ins_feature()

        shs = gaussians.get_features()
        rgb = SH2RGB(shs)
        if use_ins:
            feature = torch.cat([rgb, instance_feature], dim=-1)
        else:
            feature = rgb
        
        if use_geo:
            pts = gaussians.get_xyz()
            geo_feature = Attn.PEn(pts)
            feature = torch.cat([feature, geo_feature], dim=-1)

        D = feature.shape[-1]

        pred_lang_feat, _ = Attn.inference(feature.reshape(-1, D).float())

        # Visualize language feature map
        x_pca = pca.fit_transform(pred_lang_feat.cpu().numpy())
        feat_vis = torch.from_numpy(x_pca)
        feat_vis = (feat_vis - feat_vis.min()) / (feat_vis.max() - feat_vis.min())
        ## TODO save ply

        # Load GT point cloud and labels
        point_labels, target_names, point_cloud = load_scannet_gt(gt_file_path)

        # Convert language features into labels
        query_text_feats = F.normalize(load_query_text_features(target_names, text_feature_dir), dim=1, p=2)                   # (num_target_ids, 512)
        cosine_similarity = torch.matmul(query_text_feats, pred_lang_feat.transpose(0, 1))
        predicted_labels = torch.argmax(cosine_similarity, dim=0) + 1


        gaussians_params = {
            'mu': gaussians.get_xyz(),
            'scale': gaussians.get_ins_scaling(),
            'rotation': gaussians.get_ins_rotation(),
            'opacity': gaussians.get_ins_opacity()
        }

        results = evaluator.evaluate(
            gaussians_params, predicted_labels, point_cloud, point_labels
        )

        print(f"Evaluation completed.")
    
        print("\nVolume-aware IoU results:")
        for label, iou in results['ious'].items():
            if isinstance(label, int):  # Skip 'mean_iou' key
                print(f"Class {label}: {iou:.4f}")
        print(f"Mean IoU: {results['ious']['mean_iou']:.4f}")
        
        print("\nVolume-aware Accuracy results:")
        print(f"Overall Accuracy: {results['volume_aware_accuracy']['overall_accuracy']:.4f}")
        print(f"Mean Class Accuracy (mAcc): {results['volume_aware_accuracy']['mean_class_accuracy']:.4f}")
        
        print("\nPer-class Volume-aware Accuracies:")
        for i, acc in enumerate(results['volume_aware_accuracy']['per_class_accuracies']):
            print(f"Class {i}: {acc:.4f}")
        
        print("\nStandard Accuracy results (for comparison):")
        print(f"Standard Overall Accuracy: {results['standard_accuracy']['std_overall_accuracy']:.4f}")
        print(f"Standard mAcc: {results['standard_accuracy']['std_mean_class_accuracy']:.4f}")

            
if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Visualization script parameters")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--json_dir", type=str, default='dataset/lerf_ovs/label')
    parser.add_argument("--mask_thresh", type=float, default=0.4)
    parser.add_argument("--scene_name", type=str, default=None)
    parser.add_argument("--text_feature_dir", type=str, default='dataset/lerf_ovs/label/clip')  ##TODO
 
    op, model, pipeline = OptimizationParams(parser), ModelParams(parser, sentinel=True), PipelineParams(parser)
    args = get_combined_args(parser)
    print("[INFO]: Evaluating file " + args.model_path)
    print(f"[INFO]: {args}")
    
    # Initialize system state (RNG)
    safe_state(args.quiet)
    seed_everything(seed_value=42)

    dataset_args = model.extract(args)
    opt_args = op.extract(args)
    pipe_args = pipeline.extract(args)
    
    scene_name = args.scene_name
    gt_file_path = os.path.join(dataset_args.model_path, args.scene_name)
    text_feature_dir = args.text_feature_dir
    ckpt_path = f"{dataset_args.model_path}/ckpt30000"

    # Generate Mask for each queries
    evaluate(dataset_args, opt_args, ckpt_path, gt_file_path, text_feature_dir)
