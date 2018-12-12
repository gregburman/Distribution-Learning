# Distribution-Learning

implementation of a [Probabilistic U-net](https://arxiv.org/abs/1806.05034) on connectomic datasets. The model  produces a (specified) number of predicted plausible segmentation samples for a given input volume.

Dependencies:

* [gunpowder](https://github.com/funkey/gunpowder): required for building the data-loading pipeline.
* [skelerator](https://github.com/nilsec/skelerator): required for generating the neural toy data used for testing the implementation
* [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) (optional): required for training the model in a docker container that leverages NVIDIA GPUs
