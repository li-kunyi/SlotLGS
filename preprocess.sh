# python preprocessor/run.py --dataset_path /home/kunyi/work/data/lerf_ovs/figurines --sam_ckpt_path ckpt/sam_vit_h_4b8939.pth --encoder dinov3 #--empty_bg
# python preprocessor/run.py --dataset_path /home/kunyi/work/data/lerf_ovs/ramen --sam_ckpt_path ckpt/sam_vit_h_4b8939.pth --encoder dinov3 #--empty_bg
# python preprocessor/run.py --dataset_path /home/kunyi/work/data/lerf_ovs/teatime --sam_ckpt_path ckpt/sam_vit_h_4b8939.pth --encoder dinov3 #--empty_bg
# python preprocessor/run.py --dataset_path /home/kunyi/work/data/lerf_ovs/waldo_kitchen --sam_ckpt_path ckpt/sam_vit_h_4b8939.pth --encoder dinov3 #--empty_bg

python preprocessor/cluster_language_slot.py --dataset_path /home/kunyi/work/data/lerf_ovs/figurines/preprocess/features/clip
python preprocessor/cluster_language_slot.py --dataset_path /home/kunyi/work/data/lerf_ovs/ramen/preprocess/features/clip
python preprocessor/cluster_language_slot.py --dataset_path /home/kunyi/work/data/lerf_ovs/teatime/preprocess/features/clip
python preprocessor/cluster_language_slot.py --dataset_path /home/kunyi/work/data/lerf_ovs/waldo_kitchen/preprocess/features/clip