import objaverse
import os
import shutil
import torch
import torch.utils.data
import trimesh
import pathlib
import numpy as np
from tqdm import tqdm

from pytorch3d.io import IO
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.renderer import (
	Textures,
	TexturesUV,
	TexturesVertex,
)

class ObjaverseDataset(torch.utils.data.Dataset):
	def __init__(self, args):
		super(ObjaverseDataset, self).__init__()
		self.args = args
		self.device = args.device
		self.filelist = ObjaverseFileList(args, args.total_uid_counts, args.objaverse_dir, args.output_dir)
		self.temp_dir = args.output_dir

	def get_geometry(self, filename_obj):
		if self.args.debug and os.path.getsize(filename_obj) > 100 * 1024 * 1024:
			# skip large mesh
			return None, False
		try:
			scene = trimesh.load(filename_obj)
		except:
			return None, False
		geometry = trimesh.util.concatenate(scene.dump())
		if isinstance(geometry, trimesh.Trimesh):
			return geometry, True
		else:
			return None, False
	
	def get_textures(self, visual, verts, faces):
		material = visual.material
		# TODO: whether the mesh is valid
		maps, main_color, valid = None, None, True
		if isinstance(material, trimesh.visual.material.SimpleMaterial):
			if material.image is not None:
				maps = material.image
			else:
				main_color = material.main_color
		else:
			if material.baseColorTexture is not None:
				maps = material.baseColorTexture
			elif material.baseColorFactor is not None:
				main_color = material.baseColorFactor
			else:
				main_color = material.main_color
		if maps is not None:
			if maps.mode != 'RGB':
				maps = maps.convert('RGB')
			maps = torch.tensor(np.array(maps), dtype=torch.float)
			maps = torch.div(maps, 255.0)
			maps = maps.unsqueeze(0)
			uvs = torch.tensor(visual.uv, dtype=torch.float).unsqueeze(0)
			textures = TexturesUV(maps, faces, uvs)
		elif main_color is not None:
			vert_colors = torch.tensor(main_color, dtype=torch.float)
			vert_colors = vert_colors[:3] / 255.
			vert_colors = vert_colors.reshape(1, 1, -1).repeat(1, verts.shape[1], 1)
			textures = TexturesVertex(vert_colors)
		return textures, valid

	def load_mesh(self, filename_obj):
		if "glb" in self.args.save_file_type:
			self.copy_glb(filename_obj)
		
		geometry, valid = self.get_geometry(filename_obj)
		if not valid:
			print("Invalid geometry.", flush=True)
			return None, valid
		verts = torch.tensor(geometry.vertices, dtype=torch.float).unsqueeze(0)
		faces = torch.tensor(geometry.faces, dtype=torch.long).unsqueeze(0)
		textures, valid = self.get_textures(geometry.visual, verts, faces)
		mesh = Meshes(verts, faces, textures)

		verts = mesh.verts_packed()
		center = verts.mean(0)
		scale = max((verts - center).abs().max(0)[0])
		mesh.offset_verts_(-center)
		mesh.scale_verts_((1.0 / float(scale)))

		if not valid:
			print("Invalid texture type.", flush=True)
			return None, valid
		if "obj" in self.args.save_file_type:
			self.save_obj(filename_obj, geometry)
		return mesh, valid

	def copy_glb(self, filename_obj):
		filename = os.path.basename(filename_obj)[:-4]
		save_dir = os.path.join(self.temp_dir, f"temp/{filename}")
		os.makedirs(save_dir, exist_ok=True)
		shutil.copy2(filename_obj, os.path.join(save_dir, "origin.glb"))


	def save_obj(self, filename_obj, geometry):
		filename = os.path.basename(filename_obj)[:-4]
		save_dir = os.path.join(self.temp_dir, f"temp/{filename}")
		os.makedirs(save_dir, exist_ok=True)
		trimesh.exchange.export.export_mesh(geometry, os.path.join(save_dir, f"trimesh.obj"), file_type="obj")
		# IO().save_mesh(mesh, os.path.join(save_dir, f"pytorch3d.obj"), include_textures=True)

	def __len__(self):
		return len(self.filelist.glbs)

	def __getitem__(self, idx):
		uid = self.filelist.uids[idx]
		filename_ply = os.path.join(self.args.output_dir, "pointcloud", uid, "pointcloud.npz")
		if self.args.resume and os.path.exists(filename_ply):
			print(f"Mesh {uid} has exists.", flush=True)
			mesh, valid = None, False
		else:
			mesh, valid = self.load_mesh(self.filelist.glbs[uid])
		return {
			"mesh": mesh,
			"uid": uid,
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

		if self.args.debug:
			all_uids = all_uids[2260:]

		for uid in tqdm(all_uids):
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
		root_folder = self.output_dir
		glb_length = len(self.glbs)
		train_length = int(glb_length)
		filelist_folder = os.path.join(root_folder, 'filelist')
		if not os.path.exists(filelist_folder):
			os.makedirs(filelist_folder)
		train_list = os.path.join(filelist_folder, 'all.txt')
		# eval_list = os.path.join(filelist_folder, 'val.txt')
		filenames = list(self.glbs.keys())
		with open(train_list, "w") as f:
			for filename in filenames[:train_length]:
				f.write(filename)
				f.write('\n')
		# with open(eval_list, "w") as f:
		# 	for filename in filenames[train_length:]:
		# 		f.write(filename)
		# 		f.write('\n')