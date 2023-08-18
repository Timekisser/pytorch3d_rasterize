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

from dataset.filelist import FileList

class MeshDataset(torch.utils.data.Dataset):
	def __init__(self, args, device="cpu"):
		super(MeshDataset, self).__init__()
		self.args = args
		self.device = device
		self.filelist = FileList(args, args.total_uid_counts, args.objaverse_dir)
		self.temp_dir = args.output_dir

	def get_geometry(self, filename_obj):
		scene = trimesh.load(filename_obj)
		# with open(filename_obj, "rb") as f:
		# 	scene = trimesh.exchange.gltf.load_glb(f)
		geometry = trimesh.util.concatenate(scene.dump())
		return geometry

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
			maps = torch.tensor(np.array(maps), dtype=torch.float, device=self.device)
			maps = torch.div(maps, 255.0)
			maps = maps.unsqueeze(0)
			uvs = torch.tensor(visual.uv, dtype=torch.float, device=self.device).unsqueeze(0)
			textures = TexturesUV(maps, faces, uvs)
		elif main_color is not None:
			vert_colors = torch.tensor(main_color, dtype=torch.float, device=self.device)
			vert_colors = vert_colors[:3] / 255.
			vert_colors = vert_colors.reshape(1, 1, -1).repeat(1, verts.shape[1], 1)
			textures = TexturesVertex(vert_colors)
		return textures, valid
	
	def load_mesh(self, filename_obj):
		geometry = self.get_geometry(filename_obj)
		verts = torch.tensor(geometry.vertices, dtype=torch.float, device=self.device).unsqueeze(0)
		faces = torch.tensor(geometry.faces, dtype=torch.int, device=self.device).unsqueeze(0)
		textures, valid = self.get_textures(geometry.visual, verts, faces)
		mesh = Meshes(verts, faces, textures)
		mesh = mesh.to(self.device)

		verts = mesh.verts_packed()
		center = verts.mean(0)
		scale = max((verts - center).abs().max(0)[0])
		mesh.offset_verts_(-center)
		mesh.scale_verts_((1.0 / float(scale)))
		self.save_temp_mesh(filename_obj, geometry, mesh) 
		return mesh, valid

	def save_temp_mesh(self, filename_obj, geometry, mesh):
		filename = os.path.basename(filename_obj)[:-4]
		save_dir = os.path.join(self.temp_dir, f"temp/{filename}")
		os.makedirs(save_dir, exist_ok=True)
		
		if "glb" in self.args.save_file_type:
			shutil.copy2(filename_obj, os.path.join(save_dir, "origin.glb"))
		
		if "obj" in self.args.save_file_type:
			trimesh.exchange.export.export_mesh(geometry, os.path.join(save_dir, f"trimesh.obj"), file_type="obj")
			# IO().save_mesh(mesh, os.path.join(save_dir, f"pytorch3d.obj"), include_textures=True)

	def __len__(self):
		return len(self.filelist.glbs)
	
	def __getitem__(self, idx):
		uid = self.filelist.uids[idx]
		filename_ply = os.path.join(self.args.output_dir, "pointcloud", uid, "pointcloud.npz")
		if os.path.exists(filename_ply):
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
		if batch is not None:
			self.next_batch.record_stream(torch.cuda.current_stream())
		self.preload()
		return batch
