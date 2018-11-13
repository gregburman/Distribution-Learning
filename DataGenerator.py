import gunpowder as gp
from gunpowder.nodes.batch_provider import BatchFilter

class DataGenerator(BatchFilter):

	def __init__(self):
		pass
	
	def setup(self):
		self.provides(
			gp.ArrayKey('RAW'),
			gp.ArraySpec(
				roi=gp.Roi((0, 0, 0), (self.size, self.size, 1)),
				voxel_size=(1, 1, 1)))

	def process(self, batch, request):
		pass