SETUP_DIR=$1
ITERATIONS=$2

nvidia-docker run -v /home/greg/git/distribution-learning:/distribution-learning greggoburman/distribution-learning:latest python train/prob_unet/$SETUP_DIR/train.py $SETUP_DIR $ITERATIONS