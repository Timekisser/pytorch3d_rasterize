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


class ShapeNetDataset(torch.utils.data.Dataset):
	def __init__(self, args):
		super(ShapeNetDataset, self).__init__()
		self.args = args
		self.device = args.device
		self.output_dir = args.output_dir
		self.pointcloud_dir = os.path.join(self.output_dir, "pointcloud")
		self.mesh_dir = args.shapenet_mesh_dir
		self.filelist = ShapeNetFileList(args, args.total_uid_counts)

	def get_geometry(self, filename_obj):
		geometry = trimesh.load(filename_obj, force="mesh")
		# geometry = trimesh.util.concatenate(scene.dump())
		valid = False
		if isinstance(geometry, trimesh.Trimesh):
			valid = True
		return geometry, valid

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

	def barycentric_interpolation(self, points, values, interp_points):
		# 计算重心坐标
		v0 = interp_points - points[:, 0, :]
		v1 = interp_points - points[:, 1, :]
		v2 = interp_points - points[:, 2, :] 

		s0 = np.linalg.norm(np.cross(v1, v2, axis=1), axis=1)
		s1 = np.linalg.norm(np.cross(v2, v0, axis=1), axis=1)
		s2 = np.linalg.norm(np.cross(v0, v1, axis=1), axis=1)
		
		S = s0 + s1 + s2
		w0 = (s0 / S)[:, None]
		w1 = (s1 / S)[:, None]
		w2 = (s2 / S)[:, None]

		# 计算插值值
		interp_values = w0 * values[:, 0, :] + w1 * values[:, 1, :] + w2 * values[:, 2, :]

		return interp_values

	def gen_points(self, geometry, filename):
		filename_pts = os.path.join(self.pointcloud_dir, filename, 'pointcloud.npz')
		filename_ply = os.path.join(self.pointcloud_dir, filename, 'pointcloud.ply')
		os.makedirs(os.path.dirname(filename_pts), exist_ok=True)

		points, face_idx = trimesh.sample.sample_surface(geometry, self.args.num_points)
		normals = geometry.face_normals[face_idx]
		visual = geometry.visual
		material = visual.material
		if isinstance(material, trimesh.visual.material.SimpleMaterial):
			if material.image is not None:
				vertex_colors = material.to_color(visual.uv)  # (N, 3)
				faces = geometry.faces[face_idx]              # (F, 3)
				face_to_vertex = geometry.vertices[faces]     # (F, 3, 3)
				face_vertex_color = vertex_colors[faces]      # (F, 3, 4)
				colors = self.barycentric_interpolation(face_to_vertex, face_vertex_color, points)
				colors = colors[..., :3] / 255.0
			else:
				colors = np.tile(material.main_color.reshape(1, -1), (points.shape[0], 1))
		else:
			raise Exception("Colors Error!")
		
		if "pointcloud" in self.args.save_file_type:
			pointcloud = trimesh.PointCloud(vertices=points, colors=colors)
			pointcloud.export(filename_ply, file_type="ply")
		if "data" in self.args.save_file_type:
			np.savez(filename_pts, points=points.astype(np.float16), normals=normals.astype(np.float16), colors=colors.astype(np.float16))

	def load_mesh(self, filename):
		filename_obj = os.path.join(self.mesh_dir, filename,  'model.obj')		
		geometry, valid = self.get_geometry(filename_obj)
		if not valid:
			print("Invalid geometry type.", flush=True)
			return None, valid

		vertices = geometry.vertices
		bbmin, bbmax = vertices.min(0), vertices.max(0)
		center = (bbmin + bbmax) * 0.5
		scale = 2.0 / (bbmax - bbmin).max()
		geometry.vertices = (vertices - center) * scale

		# self.gen_points(geometry, filename)

		verts = torch.tensor(geometry.vertices, dtype=torch.float).unsqueeze(0)
		faces = torch.tensor(geometry.faces, dtype=torch.int).unsqueeze(0)
		textures, valid = self.get_textures(geometry.visual, verts, faces)
		mesh = Meshes(verts, faces, textures)

		if not valid:
			print("Invalid texture type.", flush=True)
			return None, valid
		return mesh, valid

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
			if self.args.resume == False or os.path.exists(filepath) == False:
				uids.append(filename)
		return uids

