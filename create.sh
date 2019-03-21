SETUP_DIR=$1

mkdir log/prob_unet/$SETUP_DIR/
mkdir snapshots/prob_unet/$SETUP_DIR/

mkdir train/prob_unet/$SETUP_DIR/
cp train/prob_unet/base_setup/mknet_train.py train/prob_unet/$SETUP_DIR/
cp train/prob_unet/base_setup/train.py train/prob_unet/$SETUP_DIR/
cp train/prob_unet/base_setup/mknet_predict.py train/prob_unet/$SETUP_DIR/
cp train/prob_unet/base_setup/predict.py train/prob_unet/$SETUP_DIR/
cp train/prob_unet/base_setup/predict_hdf.py train/prob_unet/$SETUP_DIR/