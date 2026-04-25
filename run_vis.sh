DATASET_PATH="/root/autodl-fs/lerf_ovs"
SAVE_PATH="output/lerf_ovs_0420_mlp"
ENCODER="clip"
LEVELS=(
  "l"
  # "m"
  # "s"
  )

SCENES=(
  # "teatime" 
  "figurines" 
  # "ramen" 
  # "waldo_kitchen"
  )

for SCENE_NAME in "${SCENES[@]}"
do
  echo "Processing scene: $SCENE_NAME"

  python -m vis.vis_slot_lerf \
    -s $DATASET_PATH/$SCENE_NAME \
    -m $SAVE_PATH/$SCENE_NAME \
    --encoder $ENCODER \
    --scene_name $SCENE_NAME \
    --gaussian_ckpt $SAVE_PATH/$SCENE_NAME \
    --attn_ckpt $SAVE_PATH/$SCENE_NAME \
    --json_dir $DATASET_PATH/label \
    --level l

  python -m vis.save_select_gaussian_lerf \
    -s $DATASET_PATH/$SCENE_NAME \
    -m $SAVE_PATH/$SCENE_NAME \
    --encoder $ENCODER \
    --scene_name $SCENE_NAME \
    --gaussian_ckpt $SAVE_PATH/$SCENE_NAME \
    --attn_ckpt $SAVE_PATH/$SCENE_NAME \
    --json_dir $DATASET_PATH/label \
    --level l

done

