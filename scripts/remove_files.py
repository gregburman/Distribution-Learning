import os
import sys
sys.path.append('../')

if __name__ == "__main__":

	setup = sys.argv[1]
	file_type = sys.argv[2]
	iterations = int(sys.argv[3])
	old_batch_size = int(sys.argv[4])
	new_batch_size = int(sys.argv[5])

	for i in range(int(iterations/old_batch_size)):
		n = i*old_batch_size
		if n % new_batch_size != 0:
			if file_type == "checkpoints":
				name = "train/prob_unet/" + setup + "/train_net_checkpoint_%i.*"%n
			elif file_type == "snapshots":
				name = "snapshots/prob_unet/" + setup + "/batch_%i.hdf"%(n+1)
			os.system("sudo rm " + name)
