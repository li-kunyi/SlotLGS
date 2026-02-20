DATASET_PATH="/root/autodl-fs/lerf_ovs"
SAVE_PATH="output/lerf_ovs_clip"
SCENE_NAME="figurines"
ENCODER="clip"

python train.py \
  -s $DATASET_PATH/$SCENE_NAME \
  -m $SAVE_PATH/$SCENE_NAME \
  --encoder $ENCODER \
  --ckpt_path $SAVE_PATH/$SCENE_NAME/ckpt15000

python -m eval.run_lerf \
  -s $DATASET_PATH/$SCENE_NAME \
  -m $SAVE_PATH/$SCENE_NAME \
  --encoder $ENCODER \
  --scene_name $SCENE_NAME \
  --gaussian_ckpt $SAVE_PATH/$SCENE_NAME/ckpt30000 \
  --attn_ckpt $SAVE_PATH/$SCENE_NAME/ckpt_attn5000 \
  --json_dir $DATASET_PATH/label \
  --text_feature_dir eval/clip

# python -m eval.run_lerf_3d \
#   -s $DATASET_PATH/$SCENE_NAME \
#   -m $SAVE_PATH/$SCENE_NAME \
#   --encoder $ENCODER \
#   --scene_name $SCENE_NAME \
#   --gaussian_ckpt $SAVE_PATH/$SCENE_NAME/ckpt30000 \
#   --attn_ckpt $SAVE_PATH/$SCENE_NAME/ckpt_attn5000 \
#   --json_dir $DATASET_PATH/label \
#   --text_feature_dir eval/clip

  python -m eval.run_lerf_3d_drsplat \
  -s $DATASET_PATH/$SCENE_NAME \
  -m $SAVE_PATH/$SCENE_NAME \
  --encoder $ENCODER \
  --scene_name $SCENE_NAME \
  --gaussian_ckpt $SAVE_PATH/$SCENE_NAME/ckpt30000 \
  --attn_ckpt $SAVE_PATH/$SCENE_NAME/ckpt_attn5000 \
  --json_dir $DATASET_PATH/label \
  --text_feature_dir eval/clip