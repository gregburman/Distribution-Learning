import numpy as np
from skelerator.forest import create_segmentation

seed = 0
np.random.seed(0)
print "seed state: ", np.random.get_state()[1][0]
for i in range(3):
	data = create_segmentation(shape=[100,100,100], n_objects=20, points_per_skeleton=8, smoothness=3, noise_strength=1, interpolation="random", seed=seed)
	print "hash: ", np.sum(data["segmentation"])
print ""

seed = 5
for i in range(3):
	data = create_segmentation(shape=[100,100,100], n_objects=20, points_per_skeleton=8, smoothness=3, noise_strength=1, interpolation="random", seed=seed)
	print "hash: ", np.sum(data["segmentation"])
print ""

for i in range(3):
	data = create_segmentation(shape=[100,100,100], n_objects=20, points_per_skeleton=8, smoothness=3, noise_strength=1, interpolation="random", seed=np.random.randint(0, 1000000))
	print "hash: ", np.sum(data["segmentation"])


