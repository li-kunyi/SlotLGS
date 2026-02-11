
import os
import sys
import json
import torch
import numpy as np
from PIL import Image
import glob
from typing import Dict, Union
from collections import defaultdict
from pathlib import Path
from eval.utils import polygon_to_mask, stack_mask, vis_mask_save


neon_colors = [
    (255,   0, 255),  # Neon Magenta
    (204, 255,   0),  # Electric Lime
    (255, 255,   0),  # Bright Yellow
    (0,   255, 255),  # Cyan
    (125, 249, 255),  # Electric Blue
    ( 57, 255,  20),  # Neon Green
    (255, 105, 180),  # Hot Pink
    (255,  95,  31),  # Bright Orange
    (255,   0,   0),  # Red
    (148,   0, 211),  # Neon Purple
    (191, 255,   0),  # Lime
    (  0, 255, 127),  # Spring Green
    (255,  20, 147),  # Deep Pink
]


def get_query_text_features(scene_name, feature_dir):
    """
    Load and return the CLIP language features associated with a given scene's textual queries.
        param: scene_name (str): Name of the scene (e.g., 'teatime', 'figurines').

    Returns:
        - target_text (List[str]): List of object/category names for the selected scene.
         query_text_feats (Dict[str, Tensor]): Dictionary mapping each object name to its 
                                               corresponding CLIP language feature tensor of shape [1, 512].

    Notes:
        - Loads a precomputed JSON file ('text_features.json') that maps each possible label to 
          its 512-dimensional CLIP feature vector.
        - Matches only the labels relevant to the given scene.
    """
    scene_texts = {
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
                'bag of cookies']
    }
    
    with open(os.path.join(feature_dir, 'text_features.json'), 'r') as f:
        data_loaded = json.load(f)
    all_texts = list(data_loaded.keys())
    text_features = torch.from_numpy(np.array(list(data_loaded.values()))).to(torch.float32)  # [num_text, 512]
    
    target_text = scene_texts[scene_name]
    query_text_feats = {}
    for i, text in enumerate(target_text):
        feat = text_features[all_texts.index(text)].unsqueeze(0)
        query_text_feats[text] = feat
    
    return target_text, query_text_feats

def get_queries(dataset_name): 
    scene_texts = {
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
                'bag of cookies']
    }
    all_queries = scene_texts[dataset_name]
    # Asigna un color neón cíclicamente
    return {
        q: neon_colors[i % len(neon_colors)]
        for i, q in enumerate(all_queries)
    }

"""
    all_queries = scene_texts[dataset_name]
    cmap = plt.get_cmap("tab20", len(all_queries))  # tab20 da hasta 20, puedes usar "hsv" si necesitas más

    # Map one colour to one query
    color_map = {}
    for idx, query in enumerate(all_queries):
        r, g, b, _ = cmap(idx)
        color_map[query] = (int(255 * r), int(255 * g), int(255 * b))
    return color_map
"""

def eval_gt_lerfdata(json_folder: Union[str, Path] = None, ouput_path: Path = None) -> Dict:
    """
    Organise lerf's gt annotations
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
    gt_json_paths = sorted(glob.glob(os.path.join(str(json_folder), 'frame_*.json'))) # [..., 'dataset/lerf_ovs/label/teatime/frame_00002.json', ...]
    img_paths = sorted(glob.glob(os.path.join(str(json_folder), 'frame_*.jpg')))      # [..., 'dataset/lerf_ovs/label/teatime/frame_00002.jpg', ...]
    gt_ann = {}
    for js_path in gt_json_paths:
        img_ann = defaultdict(dict)
        with open(js_path, 'r') as f:
            gt_data = json.load(f) 

        h, w = gt_data['info']['height'], gt_data['info']['width']
        idx = int(gt_data['info']['name'].split('_')[-1].split('.jpg')[0]) - 1  # image_name = 00002 -1 = 1
        for prompt_data in gt_data["objects"]: # For each instance object in the image idx 
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
            save_path = ouput_path / 'gt' / gt_data['info']['name'].split('.jpg')[0] / f'{label}.jpg' # Save at: {model_path}/{dataset}/gt/{image_name}/{category_label}.jpg'
            save_path.parent.mkdir(exist_ok=True, parents=True)
            vis_mask_save(mask, save_path)
        gt_ann[f'{idx}'] = img_ann

    return gt_ann, (h, w), img_paths


def load_image_as_binary(image_path, is_png=False, threshold=10):
    image = Image.open(image_path)
    if is_png:
        image = image.convert('L')
    image_array = np.array(image)
    binary_image = (image_array > threshold).astype(int)
    return binary_image

def calculate_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0
    return intersection / union

def evalute(gt_base, pred_base, scene_name, eval_dir):
    scene_gt_frames = {
        "waldo_kitchen": ["frame_00053", "frame_00066", "frame_00089", "frame_00140", "frame_00154"],
        "ramen": ["frame_00006", "frame_00024", "frame_00060", "frame_00065", "frame_00081", "frame_00119", "frame_00128"],
        "figurines": ["frame_00041", "frame_00105", "frame_00152", "frame_00195"],
        "teatime": ["frame_00002", "frame_00025", "frame_00043", "frame_00107", "frame_00129", "frame_00140"]
    }
    frame_names = scene_gt_frames[scene_name]

    ious = []
    for frame in frame_names:
        print("frame:", frame)
        gt_floder = os.path.join(gt_base, frame)
        file_names = [f for f in os.listdir(gt_floder) if f.endswith('.jpg')]
        pred_folder = os.path.join(pred_base, frame)
        for file_name in file_names:
            base_name = os.path.splitext(file_name)[0]
            gt_obj_path = os.path.join(gt_floder, file_name)
            pred_obj_path = os.path.join(pred_folder, 'mask', base_name + '.png')
            if not os.path.exists(pred_obj_path):
                print(f"Missing pred file for {file_name}, skipping...")
                print(f"IoU for {file_name}: 0")
                ious.append(0.0)
                continue
            mask_gt = load_image_as_binary(gt_obj_path)
            mask_pred = load_image_as_binary(pred_obj_path, is_png=True)
            iou = calculate_iou(mask_gt, mask_pred)
            ious.append(iou)
            print(f"IoU for {file_name} and {base_name + '.png'}: {iou:.4f}")
    
    # Acc.
    total_count = len(ious)
    count_iou_025 = (np.array(ious) > 0.25).sum()
    count_iou_05 = (np.array(ious) > 0.5).sum()

    # mIoU
    average_iou = np.mean(ious)
    acc_025 = count_iou_025/total_count
    acc_050 = count_iou_05/total_count
    print(f"Average IoU: {average_iou:.4f}")
    print(f"Acc@0.25: {acc_025:.4f}")
    print(f"Acc@0.5: {acc_050:.4f}")

    # Store in an txt file
    results_file = os.path.join(eval_dir, "eval2d_results.txt")
    with open(results_file, 'a') as f:  # Usa 'a' para añadir sin borrar lo anterior
        f.write(f"2D Evaluation Results for Lerf-Ovs Dataset :\n")
        f.write(f"Average IoU: {average_iou:.4f}\n")
        f.write(f"Acc@0.25: {acc_025:.4f}\n")
        f.write(f"Acc@0.5: {acc_050:.4f}\n")
        f.write(f"{'-'*40}\n")