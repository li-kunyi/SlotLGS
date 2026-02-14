DATASET_PATH="/root/autodl-fs/lerf_ovs"
SAVE_PATH="output/lerf_ovs"
SCENE_NAME="figurines"

python train.py \
  -s $DATASET_PATH/$SCENE_NAME \
  -m $SAVE_PATH/$SCENE_NAME \
  --encoder clip \
  --ckpt_path $SAVE_PATH/$SCENE_NAME/ckpt15000

python eval_lerf.py \
  -s $DATASET_PATH/$SCENE_NAME \
  -m $SAVE_PATH/$SCENE_NAME \
  --scene_name $SCENE_NAME \
  --gaussian_ckpt $SAVE_PATH/$SCENE_NAME/ckpt30000 \
  --attn_ckpt $SAVE_PATH/$SCENE_NAME/ckpt_semantic1500 \
  --json_dir $DATASET_PATH/label \
  --text_feature_dir eval/clip
  
