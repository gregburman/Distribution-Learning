import gunpowder as gp
from BlobGenerator import BlobGenerator
import matplotlib.pyplot as plt

if __name__ == "__main__":
	arrKey = gp.ArrayKey('RAW')
	request = gp.BatchRequest()
	request.add(arrKey, (100, 100))
	bg = BlobGenerator(100, 127, 2)
	print "foo"

	with gp.build(bg) as p:
		for i in range(1):
			x = p.request_batch(request)
			test = x.arrays[arrKey].data
			print test
			f1 = plt.figure(1)
			plt.imshow(test, cmap='nipy_spectral')

	plt.show()