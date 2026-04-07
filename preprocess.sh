python preprocessor/run.py --dataset_path /home/kunyi/work/data/lerf_ovs_nobg/figurines --sam_ckpt_path ckpt/sam_vit_h_4b8939.pth --encoder clip --empty_bg
python preprocessor/run.py --dataset_path /home/kunyi/work/data/lerf_ovs_nobg/ramen --sam_ckpt_path ckpt/sam_vit_h_4b8939.pth --encoder clip --empty_bg
python preprocessor/run.py --dataset_path /home/kunyi/work/data/lerf_ovs_nobg/teatime --sam_ckpt_path ckpt/sam_vit_h_4b8939.pth --encoder clip --empty_bg
python preprocessor/run.py --dataset_path /home/kunyi/work/data/lerf_ovs_nobg/waldo_kitchen --sam_ckpt_path ckpt/sam_vit_h_4b8939.pth --encoder clip --empty_bg

# python preprocessor/cluster_language_slot.py --dataset_path /root/autodl-fs/lerf_ovs_nobg/figurines/preprocess/features/clip
# python preprocessor/cluster_language_slot.py --dataset_path /root/autodl-fs/lerf_ovs_nobg/ramen/preprocess/features/clip
# python preprocessor/cluster_language_slot.py --dataset_path /root/autodl-fs/lerf_ovs_nobg/teatime/preprocess/features/clip
# python preprocessor/cluster_language_slot.py --dataset_path /root/autodl-fs/lerf_ovs_nobg/waldo_kitchen/preprocess/features/clip