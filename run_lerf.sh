DATASET_PATH="/root/autodl-fs/lerf_ovs_gala"
SAVE_PATH="output/lerf_ovs_0411_mlp"
ENCODER="clip"
LEVELS=(
  "l"
  "m"
  "s"
  "ins"
  )

SCENES=(
  # "figurines" 
  "ramen" 
  # "teatime" 
  # "waldo_kitchen"
  )

for SCENE_NAME in "${SCENES[@]}"
do
  echo "Processing scene: $SCENE_NAME"

  for LEVEL in "${LEVELS[@]}"
  do
    echo "Training with level: $LEVEL"

    python train.py \
      -s $DATASET_PATH/$SCENE_NAME \
      -m $SAVE_PATH/$SCENE_NAME/$LEVEL \
      --encoder $ENCODER \
      --level $LEVEL \
      --ckpt_path $SAVE_PATH/$SCENE_NAME
  done

  python -m eval.run_lerf_langsplat \
    -s $DATASET_PATH/$SCENE_NAME \
    -m $SAVE_PATH/$SCENE_NAME \
    --encoder $ENCODER \
    --scene_name $SCENE_NAME \
    --gaussian_ckpt $SAVE_PATH/$SCENE_NAME \
    --attn_ckpt $SAVE_PATH/$SCENE_NAME \
    --json_dir $DATASET_PATH/label \
    --text_feature_dir eval/clip \
    --level all

  python -m eval.run_lerf_3d_langsplat \
    -s $DATASET_PATH/$SCENE_NAME \
    -m $SAVE_PATH/$SCENE_NAME \
    --encoder $ENCODER \
    --scene_name $SCENE_NAME \
    --gaussian_ckpt $SAVE_PATH/$SCENE_NAME \
    --attn_ckpt $SAVE_PATH/$SCENE_NAME \
    --json_dir $DATASET_PATH/label \
    --text_feature_dir eval/clip \
    --level all

done