SETUP_DIR=$1

# unet
sudo rm log/prob_unet/$SETUP_DIR/* \
snapshots/prob_unet/$SETUP_DIR/* \
train/prob_unet/$SETUP_DIR/weights.meta \
train/prob_unet/$SETUP_DIR/config.json \
train/prob_unet/$SETUP_DIR/checkpoint \
train/prob_unet/$SETUP_DIR/weights_checkpoint* \
