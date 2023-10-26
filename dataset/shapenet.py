import objaverse
import os
import shutil
import torch
import torch.utils.data
import trimesh
import pathlib
import numpy as np
from tqdm import tqdm
import open3d as o3d

class ShapeNetDataset(torch.utils.data.Dataset):
	def __init__(self, args):
		super(ShapeNetDataset, self).__init__()
		self.args = args
		self.device = args.device
		self.output_dir = args.output_dir
		self.pointcloud_dir = os.path.join(self.output_dir, "pointcloud")
		self.mesh_dir = args.shapenet_mesh_dir
		self.filelist = ShapeNetFileList(args, args.total_uid_counts)

	def get_geometry(self, filename):
		filename_obj = os.path.join(self.mesh_dir, filename,  'model.obj')

		geometry = trimesh.load(filename_obj, force="mesh")
		# geometry = trimesh.util.concatenate(geometry.dump())
		valid = False
		if isinstance(geometry, trimesh.Trimesh):
			valid = True
		if "object" in self.args.save_file_type:
			self.save_obj(filename, trimesh_mesh=geometry)
		return geometry, valid


	def load_mesh(self, filename):
		geometry, valid = self.get_geometry(filename)
		if not valid:
			print("Invalid geometry type.", flush=True)
			return None, valid

		vertices = geometry.vertices
		bbmin, bbmax = vertices.min(0), vertices.max(0)
		center = (bbmin + bbmax) * 0.5
		scale = 2.0 / (bbmax - bbmin).max()
		geometry.vertices = (vertices - center) * scale

		return geometry, valid

	def save_obj(self, filename, trimesh_mesh=None, pytorch3d_mesh=None):
		if trimesh_mesh is not None:
			save_dir = os.path.join(self.output_dir, f"temp/{filename}/trimesh")
			os.makedirs(save_dir, exist_ok=True)
			trimesh.exchange.export.export_mesh(trimesh_mesh, os.path.join(save_dir, f"trimesh.obj"), file_type="obj")


	def __len__(self):
		return len(self.filelist.uids)

	def __getitem__(self, idx):
		uid = self.filelist.uids[idx]
		filename_ply = os.path.join(self.pointcloud_dir, uid, "pointcloud.npz")
		if self.args.resume and os.path.exists(filename_ply):
			print(f"Mesh {uid} has exists.", flush=True)
			mesh, valid = None, False
		else:
			mesh, valid = self.load_mesh(uid)
		return {
			"mesh": mesh,
			"uid": uid,
			"valid": valid,
		}

	def collect_fn(self, batch):
		return batch

class ShapeNetFileList:
	def __init__(self, args, total_uid_counts):
		self.args = args
		self.total_uid_counts = total_uid_counts
		self.output_dir = args.output_dir
		self.filelist_dir = args.shapenet_filelist_dir
		self.filenames = []
		for file_list in self.args.file_list:
			self.filenames += self.get_filenames(file_list)
		self.uids = self.get_uids()
		self.uids = ['02691156/10155655850468db78d106ce0a280f87']

	def get_filenames(self, filelist):
		filelist = os.path.join(self.filelist_dir, filelist)
		with open(filelist, 'r') as fid:
			lines = fid.readlines()
		filenames = [line.split()[0] for line in lines]
		return filenames

	def get_uids(self):
		uids = []
		for filename in self.filenames:
			if self.args.get_interior_points:
				filepath = os.path.join(self.output_dir, "interior", filename, "interior.npz")
			elif self.args.get_render_points:
				filepath = os.path.join(self.output_dir, "pointcloud", filename, "pointcloud.npz")
			elif self.args.mesh_repair:
				filepath = os.path.join(self.output_dir, "mesh_repair", filename, "model.obj")
			if self.args.resume == False or os.path.exists(filepath) == False:
				uids.append(filename)
		return uids

