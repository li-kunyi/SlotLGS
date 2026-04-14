import os
import cv2
import argparse
from tqdm import tqdm
import torch
from feature_extractor import FeatureExtractor


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--resolution', type=int, default=-1)
    parser.add_argument('--sam_ckpt_path', type=str, default="ckpt/sam_vit_h_4b8939.pth")
    parser.add_argument('--encoder', type=str, default="clip")
    parser.add_argument('--empty_bg', action='store_true', default=False)

    args = parser.parse_args()

    dataset_path = args.dataset_path
    if "scannet" in dataset_path.lower():
        img_folder = os.path.join(dataset_path, 'color')
    else:
        img_folder = os.path.join(dataset_path, 'images')

    data_list = sorted(os.listdir(img_folder))
    save_folder = os.path.join(dataset_path, 'preprocess')
    print(f"Running: {dataset_path}")

    # Segmentation: use SAM
    print(f"Image segmentation, method: SAM(langsplat), empty background: {args.empty_bg}")
    from sam import SAMProcessor
    sam_processor = SAMProcessor(sam_ckpt_path=args.sam_ckpt_path, device='cuda')

    WARNED = False
    for file_name in tqdm(data_list, desc="Processing files"):
        with torch.no_grad():
            image_path = os.path.join(img_folder, file_name)
            image = cv2.imread(image_path)
            orig_h, orig_w = image.shape[:2]

            if args.resolution == -1:
                if orig_h > 1080 and not WARNED:
                    print("[ INFO ] Large image detected (>1080P), rescaling to 1080P.")
                    WARNED = True
                scale = orig_h / 1080 if orig_h > 1080 else 1
            else:
                scale = orig_w / args.resolution

            new_h, new_w = int(orig_h / scale), int(orig_w / scale)
            image = cv2.resize(image, (new_w, new_h))
            # image = torch.from_numpy(image)

            os.makedirs(save_folder, exist_ok=True)
            with open(f"{save_folder}/resolution.txt", "w") as f:
                f.write(f"{new_w} {new_h}\n")

            sam_processor.process_images(image, file_name.split('.')[0], save_folder, empty_bg=args.empty_bg)

    # Feature Encoding: option: dinotxt, dinov3, clip
    print(f"Extract features, method: {args.encoder}")
    if args.encoder == 'dinotxt':
        from dinotxt import DinoTxtWrapper
        model = DinoTxtWrapper()
    elif args.encoder == 'dinov3':
        from dinov3 import DinoExtractor
        model = DinoExtractor()
    elif args.encoder == 'clip':
        from CLIP import OpenCLIPNetwork, OpenCLIPNetworkConfig
        model = OpenCLIPNetwork(OpenCLIPNetworkConfig)
    else:
        raise("WRONG ENCODER TYPE!")

    levels = [
                # 's', 
                # 'm', 
                'l',
             ]
    feature_extractor = FeatureExtractor(save_folder, model)
    for file_name in tqdm(data_list, desc="Processing files"):
        with torch.no_grad():
            for level in levels:
                feature_extractor.create_features(file_name.split('.')[0], method=args.encoder, level=level)
