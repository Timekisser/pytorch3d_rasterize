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
	TexturesUV,
	TexturesVertex,
)

from dataset.filelist import FileList

class MeshDataset(torch.utils.data.Dataset):
	def __init__(self, args, device="cpu"):
		super(MeshDataset, self).__init__()
		self.args = args
		self.device = device
		self.filelist = FileList(args.total_uid_counts)
		self.temp_dir = args.output_dir

	def get_geometry(self, filename_obj):
		scene = trimesh.load(filename_obj)
		# with open(filename_obj, "rb") as f:
		# 	scene = trimesh.exchange.gltf.load_glb(f)
		geometry = trimesh.util.concatenate(scene.dump())
		return geometry

	def get_textures(self, visual, faces):
		material = visual.material
		maps, main_color = None, None
		if isinstance(material, trimesh.visual.material.SimpleMaterial):
			if material.image is not None:
				maps = material.image
			else:
				main_color = material.main_color
		else:
			if material.baseColorTexture is not None:
				maps = material.baseColorTexture
			else:
				main_color = material.baseColorFactor
		if maps is not None:
			maps = torch.tensor(np.array(maps), dtype=torch.float, device=self.device).unsqueeze(0)[:, :, :, :3] / 255.
			uvs = torch.tensor(visual.uv, dtype=torch.float, device=self.device).unsqueeze(0)
			textures = TexturesUV(maps, faces, uvs)
		elif main_color is not None:
			vert_colors = torch.tensor(material.baseColorFactor, dtype=torch.float, device=self.device)
			textures = TexturesVertex(vert_colors)
		return textures
	
	def load_mesh(self, filename_obj):
		geometry = self.get_geometry(filename_obj)
		verts = torch.tensor(geometry.vertices, dtype=torch.float, device=self.device).unsqueeze(0)
		faces = torch.tensor(geometry.faces, dtype=torch.int, device=self.device).unsqueeze(0)
		textures = self.get_textures(geometry.visual, faces)
		mesh = Meshes(verts, faces, textures)

		verts = mesh.verts_packed()
		center = verts.mean(0)
		scale = max((verts - center).abs().max(0)[0])
		mesh.offset_verts_(-center)
		mesh.scale_verts_((1.0 / float(scale)))
		self.save_temp_mesh(filename_obj, geometry, mesh) 
		return mesh

	def save_temp_mesh(self, filename_obj, geometry, mesh):
		filename = os.path.basename(filename_obj)[:-4]
		pytorch3d_dir = os.path.join(self.temp_dir, "temp/pytorch3d")
		trimesh_dir = os.path.join(self.temp_dir, "temp/trimesh")
		os.makedirs(pytorch3d_dir, exist_ok=True)
		os.makedirs(trimesh_dir, exist_ok=True)
		shutil.copy2(filename_obj, os.path.join(pytorch3d_dir, f"{filename}.glb"))
		trimesh.exchange.export.export_mesh(geometry, os.path.join(trimesh_dir, f"{filename}.obj"), file_type="obj")
		IO().save_mesh(mesh, os.path.join(pytorch3d_dir, f"{filename}.obj"), include_textures=True)

	def __len__(self):
		return len(self.filelist.glbs)
	
	def __getitem__(self, idx):
		uid = self.filelist.uids[idx]
		mesh = self.load_mesh(self.filelist.glbs[uid])
		return mesh, uid

	def collect_fn(self, batch):
		return batch[0]

	
class DataPreFetcher:

	def __init__(self, loader, device):
		self.loader = iter(loader)
		self.device = device
		self.next_mesh = None
		self.next_uid = None
		self.stream = torch.cuda.Stream()
		self.preload()

	def to_cuda(self):
		self.next_mesh = self.next_mesh.to(self.device)


	def preload(self):
		try:
			self.next_mesh, self.next_uid = next(self.loader)
		except StopIteration:
			self.next_mesh = None
			self.next_uid = None
			return
		with torch.cuda.stream(self.stream):
			self.to_cuda()

	def next(self):
		torch.cuda.current_stream().wait_stream(self.stream)
		mesh, uid = self.next_mesh, self.next_uid
		if mesh is not None:
			self.next_mesh.record_stream(torch.cuda.current_stream())
		if uid is not None:
			self.next_uid.record_stream(torch.cuda.current_stream())
		self.preload()
		return [mesh, uid]