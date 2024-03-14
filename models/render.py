import torch
import torch.nn
import numpy as np
import os
import trimesh
import traceback
import matplotlib.pyplot as plt
import copy
import cv2
from PIL import Image
from pytorch3d.renderer import (
	look_at_view_transform,
	FoVPerspectiveCameras,
	FoVOrthographicCameras,
	RasterizationSettings,
)
from pytorch3d.renderer.opengl import MeshRasterizerOpenGL
from pytorch3d.ops.interp_face_attrs import interpolate_face_attributes
from pytorch3d.renderer.mesh.rasterizer import Fragments

class PointCloudRender(torch.nn.Module):
	def __init__(self, args, image_size=600, camera_dist=3, output_dir="data/Objaverse",  device="cuda") -> None:
		super().__init__()
		self.args = args
		self.image_size = image_size
		self.camera_dist = camera_dist
		self.batch_size = args.batch_size
		self.elevation =  [0, 0,  0,   0,   -90, 90] + [-45, -45, -45, -45, 45, 45,  45,  45]
		self.azim_angle = [0, 90, 180, 270, 0,   0]  + [45,  135, 225, 315, 45, 135, 225, 315]	
		self.num_views = len(self.elevation)
		self.num_points = args.num_points
		self.device = device
		self.error_count = 0

		# self.elevation = self.elevation * self.batch_size
		# self.azim_angle = self.azim_angle * self.batch_size
		if self.args.save_memory:
			self.cameras_list = []
			for elev, azim in zip(self.elevation, self.azim_angle):
				self.cameras_list.append(self.get_cameras([elev], [azim]))
		self.cameras = self.get_cameras(self.elevation, self.azim_angle)
		self.rasterizer = None
		self.full_transform = None
		# Output dir
		output_dir_list = []
		if "image" in self.args.save_file_type:
			self.image_dir = os.path.join(output_dir, self.args.image_folder)
			output_dir_list.append(self.image_dir)
		if "pointcloud" in self.args.save_file_type:
			self.pointcloud_dir = os.path.join(output_dir, self.args.pointcloud_folder)
			output_dir_list.append(self.pointcloud_dir)
		for dir in output_dir_list:
			os.makedirs(dir, exist_ok=True)

	def get_cameras(self, elevation, azim_angle):
	   	# Initialize the camera with camera distance, elevation, azimuth angle,
		# and image size
		scale = 1.2
		R, T = look_at_view_transform(dist=self.camera_dist, elev=elevation, azim=azim_angle, device=self.device)
		if self.args.camera_mode == "Perspective":
			cameras = FoVPerspectiveCameras(R=R, T=T, device=self.device)
		elif self.args.camera_mode == "Orthographic":
			cameras = FoVOrthographicCameras(
				R=R, T = T, device=self.device,
				max_x=scale, min_x=-scale, max_y=scale, min_y=-scale,
			)
		else:
			raise Exception("No such camera mode.")
		return cameras

	def get_bin_size(self, meshes):
		if self.args.bin_mode == "coarse":
			bin_size_face = int(2 ** np.floor(np.log2((meshes._F + 0.001) // 6) - 8))
			bin_size_image = int(2 ** max(np.ceil(np.log2(self.image_size)) - 4, 4))
			bin_size = max(bin_size_face, bin_size_image)
		else:
			bin_size = 0
		return bin_size
	
	def get_max_faces_per_bin(self, meshes):
		if self.args.bin_mode == "coarse":
			max_faces_per_bin = int(meshes._F) * 2
		else:
			max_faces_per_bin = None
		return max_faces_per_bin

	def get_rasterizer(self, meshes, cameras):
		raster_settings = RasterizationSettings(
			image_size=self.image_size,
			blur_radius=0.0,
			faces_per_pixel=self.args.faces_per_pixel,
			cull_backfaces=self.args.cull_backfaces,
			# bin_size=self.get_bin_size(meshes),
			# max_faces_per_bin=self.get_max_faces_per_bin(meshes),
		)
		# Initialize rasterizer by using a MeshRasterizer class
		rasterizer = MeshRasterizerOpenGL(
			cameras=cameras,
			raster_settings=raster_settings
		)
		return rasterizer

	# def get_shader(self):
	# 	# The textured phong shader interpolates the texture uv coordinates for
	# 	# each vertex, and samples from a texture image.
	# 	# lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])
	# 	shader = SplatterPhongShader(cameras=self.cameras, device=self.device)
	# 	# Create a mesh renderer by composing a rasterizer and a shader
	# 	return shader
	
	def render(self, meshes, cameras):
		rasterizer = self.get_rasterizer(meshes, cameras)
		fragments = rasterizer(meshes)
		# if "image" in self.args.save_file_type:
		# 	images = self.shader(fragments, meshes)
		# else:
		# 	images = None
		return fragments

	def gen_image(self, fragments, images, uid):
		valid = fragments.pix_to_face[:, :, :, 0] != -1
		images[valid.logical_not(), :] = 1.0
		alpha = torch.zeros(images.shape[:3], device=images.device).unsqueeze(-1)
		alpha[valid] = 1.0
		images = torch.cat([images, alpha], dim=-1)


		images = (images * 255.0).to(torch.uint8).cpu().numpy()
		save_dir = os.path.join(self.image_dir, uid)
		os.makedirs(save_dir, exist_ok=True)
		# print("Saved image as " + str(save_dir), flush=True)
		for i in range(self.num_views):
			elev = self.elevation[i]
			azim = self.azim_angle[i]
			filename_png = os.path.join(save_dir, f"elev{int(elev)}_azim{int(azim)}.png")
			im = Image.fromarray(images[i, ...], mode="RGBA")
			im.save(filename_png)

	def get_pixel_data(self, meshes, fragments):
		verts = meshes.verts_packed()  # (N, V, 3)
		faces = meshes.faces_packed()  # (N, F, 3)
		# texels = meshes.sample_textures(fragments)
		faces_verts = verts[faces]
		pixel_coords_in_camera = interpolate_face_attributes(
			fragments.pix_to_face, fragments.bary_coords, faces_verts
		)

		faces_normals = meshes._faces_normals_packed
		pixel_valid = fragments.pix_to_face != -1
		pixel_normals = faces_normals[fragments.pix_to_face]
		pixel_normals[pixel_valid.logical_not()] = 0
		return pixel_coords_in_camera, pixel_normals

	def gen_pointcloud(self, fragments, pixel_coords_in_camera, pixel_normals, texels, uid):
		valid_v, valid_x, valid_y = torch.where(fragments.pix_to_face[:, :, :, 0] != -1)
		if valid_v.shape[0] == 0:
			raise Exception("Empty mesh.")
		pixel_coords = pixel_coords_in_camera[valid_v, valid_x, valid_y, 0] # (P, 3)
		pixel_normals = pixel_normals[valid_v, valid_x, valid_y, 0]	# (P, 3)
		
		pixel_colors = texels[valid_v, valid_x, valid_y, :]	# (P, 3)

		# 正常来说得到的法向量norm都应该是1
		# 但有部分为0，有部分在0~1
		normals_norm = torch.norm(pixel_normals, p=2, dim=1)
		normals_valid = normals_norm > 0.9
		pixel_coords = pixel_coords[normals_valid]
		pixel_normals = pixel_normals[normals_valid]
		pixel_colors = pixel_colors[normals_valid]

		P = pixel_coords.shape[0]
		random_indices = torch.randint(0, P, size=(self.num_points, ), device=pixel_coords_in_camera.device)
		points = pixel_coords[random_indices]
		normals = pixel_normals[random_indices]
		normals = normals / torch.norm(normals, p=2, dim=1, keepdim=True)
		colors = pixel_colors[random_indices]

		# normal directions
		camera_centers = self.cameras.get_camera_center().to(pixel_coords_in_camera.device)
		camera_centers = camera_centers[valid_v][random_indices]
		camera_vectors =  camera_centers - points
		normals_negative = torch.sum(camera_vectors * normals, dim=1) < 0.0
		normals[normals_negative] = -1.0 * normals[normals_negative]

		# dilate points
		points = points + self.args.points_dilate * normals

		points = points.cpu().numpy()
		normals = normals.cpu().numpy()
		colors = colors.cpu().numpy()

		# RGB -> RGBA
		ones = np.ones((colors.shape[0], 1))
		colors_rgba = np.concatenate([colors, ones], axis=1)
		pointcloud = trimesh.points.PointCloud(vertices=points, colors=colors_rgba)

		save_dir = os.path.join(self.pointcloud_dir, uid)
		os.makedirs(save_dir, exist_ok=True)
		filename_ply = os.path.join(save_dir, "pointcloud.ply")
		filename_npy = os.path.join(save_dir, "pointcloud.npz")

		if "pointcloud" in self.args.save_file_type:
			pointcloud.export(filename_ply, file_type="ply")
		if "data" in self.args.save_file_type:
			np.savez(filename_npy, points=points.astype(np.float16), normals=normals.astype(np.float16), colors=colors.astype(np.float16))
		if "normal" in self.args.save_file_type:
			self.visualize_points_and_normals(points, normals, uid)
	
	def visualize_points_and_normals(self, points, normals, uid):
		cameras = self.cameras.get_camera_center().cpu().numpy()
		points_with_normals = np.concatenate([points + 0.01 * normals, points, cameras], axis=0)
		colors = np.concatenate([np.array([[1, 0, 0]] * points.shape[0]), np.array([[0, 1, 0]] * points.shape[0]), np.array([[0, 0, 1]] * self.num_views)], axis=0) * 255.0
		pointcloud = trimesh.points.PointCloud(vertices=points_with_normals, colors=colors)
		save_dir = os.path.join(self.pointcloud_dir, uid, f"normals.ply")
		os.makedirs(os.path.dirname(save_dir), exist_ok=True)
		pointcloud.export(save_dir, file_type="ply")


	def gen_interior_points(self, fragments, images, pixel_coords_in_camera, pixel_normals, uid):
		pix_to_face = fragments.pix_to_face
		V, H, W, F = pix_to_face.shape
		valid_v, valid_x, valid_y, first_valid_f = torch.where(pix_to_face[:, :, :, :1] != -1)
		last_pix_to_face = torch.cat([pix_to_face, -1 * torch.ones((V, H, W, 1), device=self.device)], dim=-1)
		last_valid_f = torch.argmin(last_pix_to_face, dim=-1)[valid_v, valid_x, valid_y] - 1

		first_pixel_coords = pixel_coords_in_camera[valid_v, valid_x, valid_y, first_valid_f] # (P, 3)
		last_pixel_coords = pixel_coords_in_camera[valid_v, valid_x, valid_y, last_valid_f] # (P, 3)


		P = first_pixel_coords.shape[0]
		random_indices = torch.randint(0, P, size=(self.args.num_interior_points, ), device=self.device)
		random_distances = torch.rand((self.args.num_interior_points, 1), device=self.device) * 0.9 + 0.05

		points = first_pixel_coords[random_indices] * random_distances + last_pixel_coords[random_indices] * (1 - random_distances)

		points = points.cpu().numpy()

		pointcloud = trimesh.points.PointCloud(vertices=points)
		save_dir = os.path.join(self.interior_dir, uid)
		# print("Saved interior pointcloud as " + str(save_dir), flush=True)
		os.makedirs(save_dir, exist_ok=True)
		filename_ply = os.path.join(save_dir, "interior.ply")
		filename_npy = os.path.join(save_dir, "interior.npz")

		if "pointcloud" in self.args.save_file_type:
			pointcloud.export(filename_ply, file_type="ply")
		if "data" in self.args.save_file_type:
			np.savez(filename_npy, points=points)	

	def forward(self, batch):
		# TODO: render by a batch
		for data in batch:
			mesh, uid, valid = data["mesh"], data["uid"], data["valid"]
			if not valid:
				self.error_count += 1
				print(f"Invalid {self.error_count} in mesh {uid}.", flush=True)
				continue
			print(f"Start render pointcloud of {uid}", flush=True)
			if self.args.save_memory:
				pixel_coords_in_camera, pixel_normals = [], []
				fragments = []
				meshes = copy.deepcopy(mesh).extend(1)
				for camera in self.cameras_list:
					fragment = self.render(meshes, camera)
					pixel_coord_in_camera, pixel_normal = self.get_pixel_data(meshes, fragment) 	# (N, P, K, 3)
					pixel_coords_in_camera.append(pixel_coord_in_camera)
					pixel_normals.append(pixel_normal)
					fragments.append(fragment)
				
				device = "cpu"
				fragments = Fragments(
					pix_to_face=torch.cat([x.pix_to_face for x in fragments], dim=0).to(device),
					zbuf=torch.cat([x.zbuf for x in fragments], dim=0).to(device),
					bary_coords=torch.cat([x.bary_coords for x in fragments], dim=0).to(device),
					dists=None,
				)
				pixel_coords_in_camera = torch.cat(pixel_coords_in_camera, dim=0).to(device)
				pixel_normals = torch.cat(pixel_normals, dim=0).to(device)
				meshes = mesh.to(device).extend(self.num_views)
			else:
				meshes = mesh.extend(self.num_views)
				fragments = self.render(meshes, self.cameras)
				pixel_coords_in_camera, pixel_normals = self.get_pixel_data(meshes, fragments) 	# (N, P, K, 3)
			
			texels = meshes.sample_textures(fragments).squeeze(-2)	
			if "image" in self.args.save_file_type:
				self.gen_image(fragments, texels, uid)
			if "pointcloud" in self.args.save_file_type:
				self.gen_pointcloud(fragments, pixel_coords_in_camera, pixel_normals, texels, uid)

