SETUP_DIR=$1

# unet
sudo rm log/prob_unet/$SETUP_DIR/* \
snapshots/prob_unet/$SETUP_DIR/* \
train/prob_unet/$SETUP_DIR/*.meta \
train/prob_unet/$SETUP_DIR/*.json \
train/prob_unet/$SETUP_DIR/*checkpoint* \
