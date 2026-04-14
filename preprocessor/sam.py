import os
import random

import cv2
import numpy as np
import torch
from PIL import Image
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry


class SAMProcessor:
    def __init__(self, sam_ckpt_path: str, device: str = 'cuda', seed: int = 42):
        self.device = device
        self.seed_everything(seed)

        self.sam = sam_model_registry["vit_h"](checkpoint=sam_ckpt_path).to(self.device)
        self.mask_generator = SamAutomaticMaskGenerator(
            model=self.sam,
            points_per_side=32,
            pred_iou_thresh=0.7,
            box_nms_thresh=0.7,
            stability_score_thresh=0.85,
            crop_n_layers=1,
            crop_n_points_downscale_factor=1,
            min_mask_region_area=100,
        )
        self.mask_generator.predictor.model.to(self.device)


    @staticmethod
    def seed_everything(seed_value: int):
        random.seed(seed_value)
        np.random.seed(seed_value)
        torch.manual_seed(seed_value)
        os.environ['PYTHONHASHSEED'] = str(seed_value)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed_value)
            torch.cuda.manual_seed_all(seed_value)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = True


    def process_images(self, img, name, root_folder, empty_bg=True):
        try:
            seg_images, seg_maps = self.sam_encoder(img, empty_bg)
        except Exception as e:
            print(f"[WARNING] Failed to process image {name}: {e}")
            return

        # save all segmentations (seg_maps: image size, seg_images: resiezed)
        save_folder = os.path.join(root_folder, 'SAM')
        os.makedirs(save_folder, exist_ok=True)
        save_path = os.path.join(save_folder, name + '.npy')
        np.save(save_path, {
            'seg_images': {k: v.cpu().numpy() for k, v in seg_images.items()},  # d/s/m/l: [N, 3, 224, 224]
            'seg_maps': seg_maps  # d/s/m/l: [H, W]
        })  # data = np.load(save_path, allow_pickle=True).item()

        # visualize each level segmentation result
        save_folder = os.path.join(root_folder, 'SAM_vis')
        os.makedirs(save_folder, exist_ok=True)
        for level, mask_map in seg_maps.items():
            save_path = os.path.join(save_folder, name + f'_{level}.png')
            self.visualize_seg_map(mask_map, save_path)

        # img_save_dir = os.path.join(root_folder, 'tiles', name)
        # os.makedirs(img_save_dir, exist_ok=True)
        # tiles_cpu = seg_images['l'].cpu()
        # for i in range(tiles_cpu.shape[0]):
        #     img = tiles_cpu[i]  # [C, H, W]
        #     if img.max() <= 1.0:
        #         img = img * 255.0
        #     img = img.clamp(0, 255).byte()
        #     img = img.permute(1, 2, 0).numpy()

        #     Image.fromarray(img).save(os.path.join(img_save_dir, f"{name}_tile_{i}.png"))


    def visualize_seg_map(self, seg_map, save_path, bg_color=(0, 0, 0), seed=0):
        H, W = seg_map.shape
        vis = np.zeros((H, W, 3), dtype=np.uint8)

        vis[seg_map == -1] = bg_color

        rng = np.random.default_rng(seed)

        ids = np.unique(seg_map)
        ids = ids[ids >= 0]

        color_map = {}
        for i in ids:
            color_map[i] = rng.integers(0, 256, size=3, dtype=np.uint8)

        for i, color in color_map.items():
            vis[seg_map == i] = color

        cv2.imwrite(save_path, vis)


    def sam_encoder(self, image, empty_bg=True):
        image_np = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2RGB)
        masks_default, masks_s, masks_m, masks_l = self.mask_generator.generate(image_np)
        masks_default, masks_s, masks_m, masks_l = self.masks_update(
            masks_default, masks_s, masks_m, masks_l, iou_thr=0.8, score_thr=0.7, inner_thr=0.5
        )

        seg_images, seg_maps = {}, {}
        # seg_images['default'], seg_maps['default'] = self.mask2segmap(masks_default, image_np, empty_bg)
        # if len(masks_s) != 0:
        #     seg_images['s'], seg_maps['s'] = self.mask2segmap(masks_s, image_np, empty_bg)
        # if len(masks_m) != 0:
        #     seg_images['m'], seg_maps['m'] = self.mask2segmap(masks_m, image_np, empty_bg)
        if len(masks_l) != 0:
            seg_images['l'], seg_maps['l'] = self.mask2segmap(masks_l, image_np, empty_bg)
            
        return seg_images, seg_maps


    def mask2segmap(self, masks, image, empty_bg=True):
        seg_img_list = []
        seg_map = -np.ones(image.shape[:2], dtype=np.int32)

        for i, mask in enumerate(masks):
            if empty_bg:
                seg_img = self.get_seg_img(mask, image)
            else:
                seg_img = self.get_seg_img_square(mask, image)

            pad_seg_img = cv2.resize(self.pad_img(seg_img), (224, 224))
            
            seg_img_list.append(pad_seg_img)
            seg_map[mask['segmentation']] = i

        seg_imgs = np.stack(seg_img_list, axis=0)
        seg_imgs = torch.from_numpy(seg_imgs.astype("float32")).permute(0, 3, 1, 2) / 255.0
        return seg_imgs.to(self.device), seg_map


    def masks_update(self, *args, **kwargs):
        masks_new = ()
        for masks_lvl in args:
            if len(masks_lvl) == 0:
                masks_new += ([],)
                continue

            seg_pred = torch.from_numpy(np.stack([m['segmentation'] for m in masks_lvl]))
            iou_pred = torch.from_numpy(np.stack([m['predicted_iou'] for m in masks_lvl]))
            stability = torch.from_numpy(np.stack([m['stability_score'] for m in masks_lvl]))

            scores = stability * iou_pred
            keep_mask_nms = self.mask_nms(seg_pred, scores, **kwargs)
            masks_lvl = self.filter_masks(keep_mask_nms, masks_lvl)
            masks_new += (masks_lvl,)
        return masks_new

    @staticmethod
    def apply_erosion(mask, kernel_size=3, iterations=1):
        eroded_mask = torch.zeros_like(mask)
        
        # Define the erosion kernel
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask_np = mask.cpu().numpy().astype(np.uint8)  # Convert to NumPy array
        eroded_mask_np = cv2.erode(mask_np, kernel, iterations=iterations)
        eroded_mask = torch.from_numpy(eroded_mask_np).bool()  # Convert back to torch.Tensor

        return eroded_mask

    @staticmethod
    def mask_nms(masks, scores, iou_thr=0.7, score_thr=0.1, inner_thr=0.2):
        scores, idx = scores.sort(descending=True)
        masks = masks[idx]
        masks_area = masks.sum(dim=(1, 2), dtype=torch.float)
        num_masks = len(masks)

        iou_matrix = torch.zeros((num_masks, num_masks), dtype=torch.float)
        inner_iou_matrix = torch.zeros((num_masks, num_masks), dtype=torch.float)

        for i in range(num_masks):
            for j in range(i, num_masks):
                intersection = torch.sum(torch.logical_and(masks[i], masks[j]), dtype=torch.float)
                union = torch.sum(torch.logical_or(masks[i], masks[j]), dtype=torch.float)
                iou = intersection / union
                iou_matrix[i, j] = iou

        iou_matrix.triu_(1)
        iou_max, _ = iou_matrix.max(dim=0)

        keep = (iou_max <= iou_thr) & (scores > score_thr)
        return idx[keep]

    @staticmethod
    def get_seg_img(mask, image):
        img = image.copy()
        img[mask['segmentation'] == 0] = 0
        x, y, w, h = np.int32(mask['bbox'])
        return img[y:y + h, x:x + w]
    
    @staticmethod
    def get_seg_img_square(mask, image):
        img = image.copy()
        H, W = img.shape[:2]

        x, y, w, h = np.int32(mask['bbox'])

        long_side = max(w, h)
        cx = x + w // 2
        cy = y + h // 2
        x1 = cx - long_side // 2
        y1 = cy - long_side // 2
        x2 = x1 + long_side
        y2 = y1 + long_side        

        crop_x1 = max(0, x1)
        crop_y1 = max(0, y1)
        crop_x2 = min(W, x2)
        crop_y2 = min(H, y2)

        if x1 < 0 or y1 < 0 or x2 >= W or y2 >= H:
            cropped = img[y:y + h, x:x + w]
        else:
            cropped = img[crop_y1:crop_y2, crop_x1:crop_x2]

        return cropped

    @staticmethod
    def pad_img(img):
        h, w, _ = img.shape
        l = max(h, w)
        pad = np.zeros((l, l, 3), dtype=np.uint8)
        if h > w:
            pad[:, (h - w) // 2:(h - w) // 2 + w, :] = img
        else:
            pad[(w - h) // 2:(w - h) // 2 + h, :, :] = img
        return pad

    @staticmethod
    def filter_masks(keep_idx, masks_result):
        return [masks_result[i] for i in keep_idx.cpu().numpy()]


