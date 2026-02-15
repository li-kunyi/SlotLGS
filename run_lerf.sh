DATASET_PATH="/root/autodl-fs/lerf_ovs"
SAVE_PATH="output/lerf_ovs"
SCENE_NAME="figurines"

# python train.py \
#   -s $DATASET_PATH/$SCENE_NAME \
#   -m $SAVE_PATH/$SCENE_NAME \
#   --encoder dinov3 \
#   --ckpt_path $SAVE_PATH/$SCENE_NAME/ckpt15000

python -m eval.run_lerf \
  -s $DATASET_PATH/$SCENE_NAME \
  -m $SAVE_PATH/$SCENE_NAME \
  --encoder dinov3 \
  --scene_name $SCENE_NAME \
  --gaussian_ckpt $SAVE_PATH/$SCENE_NAME/ckpt30000 \
  --attn_ckpt $SAVE_PATH/$SCENE_NAME/ckpt_semantic5000 \
  --json_dir $DATASET_PATH/label \
  --text_feature_dir eval/talk2dino
  
