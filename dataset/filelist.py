import objaverse
import os

class FileList:
	def __init__(self, total_uid_counts=10, total_category_counts=1):
		objaverse._VERSIONED_PATH = "/mnt/sdc/weist/objaverse/"
		self.total_uid_counts = total_uid_counts
		self.total_category_counts = total_category_counts

		self.lvis_annotations = objaverse.load_lvis_annotations()
		self.object_paths = objaverse._load_object_paths()
		self.base_dir = objaverse._VERSIONED_PATH
		self.uids = []
		self.annotations = []

		self.get_glbs()
		
	def get_glbs(self):
		self.uids = []
		for category, cat_uids in self.lvis_annotations.items():
			for uid in cat_uids:
				filepath = self.object_paths[uid]
				full_path = os.path.join(self.base_dir, filepath)
				if os.path.exists(full_path):
					self.uids.append(uid)
				if len(self.uids) >= self.total_uid_counts:
					break
		# self.annotations = objaverse.load_annotations(self.uids)
		processes = 1 #mp.cpu_count()
		self.glbs = objaverse.load_objects(self.uids, processes)

	def get_filelists(self):
		project_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
		root_folder = os.path.join(project_folder, 'data/Objaverse')
		glb_length = len(self.glbs)
		train_length = int(glb_length * 0.8)
		filelist_folder = os.path.join(root_folder, 'filelist')
		if not os.path.exists(filelist_folder):
			os.makedirs(filelist_folder)
		train_list = os.path.join(filelist_folder, 'train.txt')
		eval_list = os.path.join(filelist_folder, 'val.txt')
		filenames = list(self.glbs.keys())
		with open(train_list, "w") as f:
			for filename in filenames[:train_length]:
				f.write(filename)
				f.write('\n')
		with open(eval_list, "w") as f:
			for filename in filenames[train_length:]:
				f.write(filename)
				f.write('\n')