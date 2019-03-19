SETUP_DIR=$1
CHECKPOINT=$2
SAMPLES=$3

nvidia-docker run -v /home/greg/git/distribution-learning:/distribution-learning greggoburman/distribution-learning:latest python train/prob_unet/$SETUP_DIR/predict_hdf.py $SETUP_DIR $CHECKPOINT $SAMPLES