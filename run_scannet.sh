DATASET_PATH="/root/autodl-fs/scannet_ovs"
SAVE_PATH="output/scannet_0421_mlp"
ENCODER="clip"
LEVELS=(
  "l"
  )

SCENES=(
  "scene0000_00" 
  "scene0062_00" 
  "scene0070_00" 
  "scene0097_00"
  "scene0140_00"
  "scene0347_00"
  "scene0590_00"
  "scene0645_00"
  )

for SCENE_NAME in "${SCENES[@]}"
do
  echo "Processing scene: $SCENE_NAME"

  # for LEVEL in "${LEVELS[@]}"
  # do
  #   echo "Training with level: $LEVEL"

  #   python train.py \
  #     -s $DATASET_PATH/$SCENE_NAME \
  #     -m $SAVE_PATH/$SCENE_NAME \
  #     --encoder $ENCODER \
  #     --level $LEVEL \
  #     --ckpt_path $SAVE_PATH/$SCENE_NAME \
  #     --margin 10 \
  #     -r 2
  # done

  python -m eval.run_scannet \
    -s $DATASET_PATH/$SCENE_NAME \
    -m $SAVE_PATH/$SCENE_NAME \
    --encoder $ENCODER \
    --scene_name $SCENE_NAME \
    --gaussian_ckpt $SAVE_PATH/$SCENE_NAME \
    --attn_ckpt $SAVE_PATH/$SCENE_NAME \
    --json_dir $DATASET_PATH/label \
    --level l \
    --text_feature_dir eval/clip/text_features.json

done