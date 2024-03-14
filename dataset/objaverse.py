import objaverse
import os
import shutil
import torch
import torch.utils.data
import trimesh
import pathlib
import random
import numpy as np
from tqdm import tqdm

from pytorch3d.structures import Meshes 
from pytorch3d.renderer import (
	TexturesUV,
	TexturesVertex,
)

class ObjaverseDataset(torch.utils.data.Dataset):
	def __init__(self, args):
		super(ObjaverseDataset, self).__init__()
		self.args = args
		self.device = args.device
		self.filelist = ObjaverseFileList(args, args.total_uid_counts, args.objaverse_dir, args.output_dir)
		self.output_dir = args.output_dir

	def get_geometry(self, filename_obj):
		if os.path.getsize(filename_obj) > 50 * 1024 * 1024:
			print("Too large mesh.", flush=True)
			return None, False
		try:
			scene = trimesh.load(filename_obj)
		except:
			print("Trimesh load mesh error.", flush=True)
			return None, False
		geometry = trimesh.util.concatenate(scene.dump())
		# if len(geometry.vertices) > 500000:
		# 	print(f"Too many face {len(geometry.vertices)}.", flush=True)
		# 	return None, False
		valid = False
		if isinstance(geometry, trimesh.Trimesh):
			valid = True
		return geometry, valid
	
	def get_textures(self, visual, verts, faces):
		# TODO: whether the mesh is valid
		maps, main_color, vertex_colors, valid = None, None, None, True
		if isinstance(visual, trimesh.visual.color.ColorVisuals):
			vertex_colors = visual.vertex_colors
		elif isinstance(visual.material, trimesh.visual.material.SimpleMaterial):
			if visual.material.image is not None:
				maps = visual.material.image
			else:
				main_color = visual.material.main_color
		else:
			if visual.material.baseColorTexture is not None:
				maps = visual.material.baseColorTexture
			elif visual.material.baseColorFactor is not None:
				main_color = visual.material.baseColorFactor
			else:
				main_color = visual.material.main_color
		if maps is not None:
			if maps.mode != 'RGB':
				maps = maps.convert('RGB')
			maps = torch.tensor(np.array(maps), dtype=torch.float)
			maps = torch.div(maps, 255.0)
			maps = maps.unsqueeze(0)
			uvs = torch.tensor(visual.uv, dtype=torch.float).unsqueeze(0)
			textures = TexturesUV(maps, faces, uvs)
		elif vertex_colors is not None:
			vert_colors = torch.tensor(vertex_colors, dtype=torch.float)
			vert_colors = vert_colors[:, :3] / 255.
			vert_colors = vert_colors.reshape(1, -1, 3)
			textures = TexturesVertex(vert_colors)	
		elif main_color is not None:
			vert_colors = torch.tensor(main_color, dtype=torch.float)
			vert_colors = vert_colors[:3] / 255.
			vert_colors = vert_colors.reshape(1, 1, -1).repeat(1, verts.shape[1], 1)
			textures = TexturesVertex(vert_colors)
		return textures, valid

	def load_mesh(self, filename_obj):
		if "origin" in self.args.save_file_type:
			self.copy_glb(filename_obj)
		
		geometry, valid = self.get_geometry(filename_obj)
		if not valid:
			return None, valid
		vertices = geometry.vertices
		bbmin, bbmax = vertices.min(0), vertices.max(0)
		center = (bbmin + bbmax) * 0.5
		scale = 2.0 / (bbmax - bbmin).max()
		geometry.vertices = (vertices - center) * scale

		verts = torch.tensor(geometry.vertices, dtype=torch.float).unsqueeze(0)
		faces = torch.tensor(geometry.faces, dtype=torch.long).unsqueeze(0)
		textures, valid = self.get_textures(geometry.visual, verts, faces)
		mesh = Meshes(verts, faces, textures)
		mesh._faces_normals_packed = torch.tensor(geometry.face_normals)
		if not valid:
			print("Invalid texture type.", flush=True)
			return None, valid

		if "object" in self.args.save_file_type:
			self.save_obj(filename_obj, trimesh_mesh=geometry, pytorch3d_mesh=mesh)

		return mesh, valid


	def copy_glb(self, filename_obj):
		filename = os.path.basename(filename_obj)[:-4]
		save_dir = os.path.join(self.output_dir, f"temp/{filename}")
		os.makedirs(save_dir, exist_ok=True)
		shutil.copy2(filename_obj, os.path.join(save_dir, "origin.glb"))

	def save_obj(self, filename_obj, trimesh_mesh=None, pytorch3d_mesh=None):
		filename = os.path.basename(filename_obj)[:-4]
		if trimesh_mesh is not None:
			save_dir = os.path.join(self.output_dir, f"temp/{filename}/trimesh")
			os.makedirs(save_dir, exist_ok=True)
			trimesh.exchange.export.export_mesh(trimesh_mesh, os.path.join(save_dir, f"trimesh.obj"), file_type="obj")

	def __len__(self):
		return len(self.filelist.glbs)

	def __getitem__(self, idx):
		uid = self.filelist.uids[idx]
		extend_uid = self.filelist.glbs[uid].split("/")[4] + "/" + uid
		filename_pointcloud = os.path.join(self.args.output_dir, self.args.pointcloud_folder, uid, "pointcloud.npz")
		if self.args.resume and os.path.exists(filename_pointcloud):
			# print(f"Mesh {uid} has exists.", flush=True)
			mesh, valid = None, False
		else:
			if self.args.debug:
				mesh, valid = self.load_mesh(self.filelist.glbs[uid])
			else:
				try:
					mesh, valid = self.load_mesh(self.filelist.glbs[uid])
				except:
					import traceback
					print(f"Load mesh {uid} error!")
					print(traceback.format_exc(), flush=True)
					mesh, valid = None, False
		if mesh is not None and mesh._F == 0:
			print("Empty mesh.", flush=True)
			mesh, valid = None, False
		return {
			"mesh": mesh,
			"uid": extend_uid,
			"valid": valid,
		}

	def collect_fn(self, batch):
		return batch


class ObjaverseFileList:
	def __init__(self, args, total_uid_counts=10, objaverse_dir="/mnt/sdc/weist/objaverse", output_dir="data/Objaverse"):
		self.args = args
		objaverse._VERSIONED_PATH = objaverse_dir 
		self.total_uid_counts = total_uid_counts

		self.lvis_annotations = objaverse.load_lvis_annotations()
		self.object_paths = objaverse._load_object_paths()
		self.output_dir = output_dir
		self.base_dir = objaverse_dir
		self.uids = []
		self.annotations = []

		self.get_glbs()
		self.get_filelists()

	def get_glbs(self):
		self.uids = []
		all_uids = []
		if self.args.have_category:
			for category, cat_uids in self.lvis_annotations.items():
				all_uids += cat_uids
		else:
			all_uids = objaverse.load_uids()
		# all_uids = ["87871d0522c9409f8e4012489764e793"]
		
		with open(os.path.join(self.args.log_dir, "error_uids.txt"), "r+") as f:
			error_uids = f.read().splitlines()
		# random.shuffle(all_uids)
		all_uids = set(all_uids)
		error_uids = set(error_uids)
		all_uids -= error_uids

		exist_count = 0
		for uid in tqdm(all_uids):
			filepath = self.object_paths[uid]
			glb_path = os.path.join(self.base_dir, filepath)
			pointcloud_path = os.path.join(self.output_dir, self.args.pointcloud_folder, uid[0], uid, "pointcloud.npz")
			if os.path.exists(glb_path):
				if os.path.exists(pointcloud_path):
					exist_count += 1
				if not (self.args.resume and os.path.exists(pointcloud_path)):
					self.uids.append(uid)
			if len(self.uids) >= self.total_uid_counts:
				break
		# self.annotations = objaverse.load_annotations(self.uids)
		# self.uids = all_uids
		print(f"{exist_count} files exist.", flush=True)
		processes = 24 #mp.cpu_count()
		self.glbs = objaverse.load_objects(self.uids, processes)

	def get_filelists(self):
		root_folder = self.output_dir
		glb_length = len(self.glbs)
		train_length = int(glb_length * 0.9)
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