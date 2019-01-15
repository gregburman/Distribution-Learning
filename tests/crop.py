import numpy as np
import gunpowder as gp
import logging

# logging.basicConfig(level=logging.DEBUG)

class TestSourceCrop(gp.BatchProvider):

    def setup(self):

        self.provides(
            gp.ArrayKey("RAW"),
            gp.ArraySpec(
                roi=gp.Roi((0, 0, 0), (100, 100, 100)),
                voxel_size=(1, 1, 1)))

    def provide(self, request):
		
		batch = gp.Batch()
		for (array_key, request_spec) in request.array_specs.items():
			array_spec = self.spec[array_key].copy()
			array_spec.roi = request_spec.roi
			print "array_spec: ", array_spec.roi.get_shape()
			data = np.zeros((array_spec.roi.get_shape()))
			batch.arrays[array_key] = gp.Array(data, array_spec)
		return batch

if __name__ == "__main__":

	raw = gp.ArrayKey("RAW")

	crop_roi = gp.Roi((0, 0, 0), (50, 50, 50))

	pipeline = (
		TestSourceCrop() + 
		gp.Crop(
			key=raw,
			fraction_negative=(0.25, 0.25, 0.25),
			fraction_positive=(0.25, 0.25, 0.25)))

	# pipeline = (TestSourceCrop())

	request = gp.BatchRequest()
	# request.add(raw, (100,100,100))
	request.add(raw, (50,50,50))


	print "request: ", request

with gp.build(pipeline) as p:
	for i in range(1):
		print "req: ", p.request_batch(request)
		req = p.request_batch(request)

		cropped = req.arrays[raw].data
		print "data: ", np.shape(cropped)