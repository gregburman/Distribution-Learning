# Distribution-Learning

implementation of a [Probabilistic U-net](https://arxiv.org/abs/1806.05034) on connectomic datasets. The model  produces a (specified) number of predicted plausible segmentation samples for a given input volume.

Dependencies:

* [gunpowder](https://github.com/funkey/gunpowder): required for building the data-loading pipeline.
* [skelerator](https://github.com/nilsec/skelerator): required for generating the neural toy data used for testing the implementation
* [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) (optional): required for training the model in a docker container that leverages NVIDIA GPUs

To generate a set of data:

1. Go into data/, run `mkdir datasets`
2. Run `python generate_full_samples n`, where `n` is the number of data samples.

To run a setup:

1. In the base project directory, run `./create.sh setup_X`, where X is the setup name/number. This creates the relevant sub directories for the setup in train/, log/ and snapshots/
2. If wanting to use a previous result, copy the 5 python files from it into your new setup
3. Run `./mknet.sh setup_X` from the base project directory to generate the meta and config files for the setup
4. Run `./train.sh setup_X n` where n is the number of iterations