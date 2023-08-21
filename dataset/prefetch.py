
class DataPreFetcher:

	def __init__(self, loader, device):
		self.loader = iter(loader)
		self.device = device
		self.next_batch = None
		self.stream = torch.cuda.Stream()
		self.preload()


	def preload(self):
		try:
			self.next_batch = next(self.loader)
		except StopIteration:
			self.next_batch = None
			return
		# with torch.cuda.stream(self.stream):
		# 	self.to_cuda()

	def next(self):
		torch.cuda.current_stream().wait_stream(self.stream)
		batch = self.next_batch
		# if batch is not None:
			# self.next_batch.record_stream(torch.cuda.current_stream())
		self.preload()
		return batch