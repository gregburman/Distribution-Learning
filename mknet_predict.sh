SETUP_DIR=$1

nvidia-docker run -v /home/greg/git/distribution-learning:/distribution-learning greggoburman/distribution-learning:latest python train/prob_unet/$SETUP_DIR/mknet_predict.py $SETUP_DIR