import numpy as np
import h5py

from scipy.stats import energy_distance

file = h5py.File('../snapshots/prob_unet/test_sample.hdf', 'r')