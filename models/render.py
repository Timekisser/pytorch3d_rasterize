import torch
import torch.nn
import numpy as np
import os
import trimesh
import sys
import matplotlib.pyplot as plt
from pytorch3d.io import IO
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.renderer import (
	look_at_view_transform,
	FoVPerspectiveCameras,
	FoVOrthographicCameras,
	Materials,
	PointLights,
	RasterizationSettings,
	MeshRenderer,
	MeshRasterizer,
	SoftPhongShader,
	PointsRenderer,
)
from pytorch3d.ops.interp_face_attrs import interpolate_face_attributes
from pytorch3d.renderer.cameras import try_get_projection_transform
class PointCloudRender(torch.nn.Module):
	def __init__(self, args, image_size=1024, camera_dist=3, output_dir="data/Objaverse",  device="cuda") -> None:
		super().__init__()
		self.args = args
		self.image_size = image_size
		self.camera_dist = camera_dist
		self.batch_size = args.batch_size
		self.elevation =  [0, 0,  0,   0,   -90, 90]
		self.azim_angle = [0, 90, 180, 270, 0,   0]
		self.num_views = len(self.elevation)
		self.num_points = args.num_points
		self.device = device

		# self.elevation = self.elevation * self.batch_size
		# self.azim_angle = self.azim_angle * self.batch_size

		self.renderer = self.get_renderer()
		self.full_transform = None
		self.cameras = None
		# Output dir
		self.image_dir = os.path.join(output_dir, "image")
		self.pointcloud_dir = os.path.join(output_dir, "pointcloud")
		self.interior_dir = os.path.join(output_dir, "interior")
		for dir in [self.image_dir, self.pointcloud_dir, self.interior_dir]:
			os.makedirs(dir, exist_ok=True)

	def get_transform(self, cameras):

		world_to_view_transform = cameras.get_world_to_view_transform()
		to_ndc_transform = cameras.get_ndc_camera_transform()
		projection_transform = try_get_projection_transform(cameras)
		if projection_transform is not None:
			projection_transform = projection_transform.compose(to_ndc_transform)
			full_transform = world_to_view_transform.compose(projection_transform)
		else:
			# Call transform_points instead of explicitly composing transforms to handle
			# the case, where camera class does not have a projection matrix form.
			full_transform = cameras.get_full_projection_transform()
		return full_transform

	def get_renderer(self):
	   	# Initialize the camera with camera distance, elevation, azimuth angle,
		# and image size
		R, T = look_at_view_transform(dist=self.camera_dist, elev=self.elevation, azim=self.azim_angle, device=self.device)
		if self.args.camera_mode == "Perspective":
			self.cameras = FoVPerspectiveCameras(R=R, T=T, device=self.device)
		elif self.args.camera_mode == "Orthographic":
			self.cameras = FoVOrthographicCameras(R=R, T = T, device=self.device)
		else:
			raise Exception("No such camera mode.")

		# self.transform = self.get_transform(cameras)

		raster_settings = RasterizationSettings(
			image_size=self.image_size,
			blur_radius=0.0,
			faces_per_pixel=self.args.faces_per_pixel,
			bin_size=None if self.args.bin_mode == "coarse" else 0,
		)
		# Initialize rasterizer by using a MeshRasterizer class
		rasterizer = MeshRasterizer(
			cameras=self.cameras,
			raster_settings=raster_settings
		)
		# The textured phong shader interpolates the texture uv coordinates for
		# each vertex, and samples from a texture image.
		# lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])
		shader = SoftPhongShader(cameras=self.cameras, device=self.device)
		# Create a mesh renderer by composing a rasterizer and a shader
		renderer = MeshRenderer(rasterizer, shader)
		return renderer

	def render(self, meshes):
		fragments = self.renderer.rasterizer(meshes)
		images = self.renderer.shader(fragments, meshes)
		return fragments, images

	def gen_image(self, images, uid):
		save_dir = os.path.join(self.image_dir, uid)
		os.makedirs(save_dir, exist_ok=True)
		print("Saved image as " + str(save_dir), flush=True)
		for i in range(self.num_views):
			elev = self.elevation[i]
			azim = self.azim_angle[i]
			filename_png = os.path.join(save_dir, f"elev{int(elev)}_azim{int(azim)}.png")
			plt.imshow(images[i, ..., :3].cpu().numpy())
			plt.savefig(filename_png)
			plt.cla()

	def get_pixel_data(self, meshes, fragments, image):
		verts = meshes.verts_packed()  # (N, V, 3)
		faces = meshes.faces_packed()  # (N, F, 3)
		# texels = meshes.sample_textures(fragments)
		vertex_normals = meshes.verts_normals_packed()  # (N, V, 3)
		faces_verts = verts[faces]
		faces_normals = vertex_normals[faces]
		pixel_coords_in_camera = interpolate_face_attributes(
			fragments.pix_to_face, fragments.bary_coords, faces_verts
		)
		pixel_normals = interpolate_face_attributes(
			fragments.pix_to_face, fragments.bary_coords, faces_normals
		)
		return pixel_coords_in_camera, pixel_normals

	def gen_pointcloud(self, fragments, images, pixel_coords_in_camera, pixel_normals, uid):
		valid_v, valid_x, valid_y = torch.where(fragments.pix_to_face[:, :, :, 0] != -1)
		pixel_coords = pixel_coords_in_camera[valid_v, valid_x, valid_y, 0] # (P, 3)
		pixel_normals = pixel_normals[valid_v, valid_x, valid_y, 0]	# (P, 3)
		pixel_colors = images[valid_v, valid_x, valid_y, :]	# (P, 4)

		P = pixel_coords.shape[0]
		random_indices = torch.randperm(P, device=self.device)[:self.num_points]
		points = pixel_coords[random_indices]
		normals = pixel_normals[random_indices]
		colors = pixel_colors[random_indices]

		points = points.cpu().numpy()
		normals = normals.cpu().numpy()
		colors = colors.cpu().numpy()

		pointcloud = trimesh.points.PointCloud(vertices=points, colors=colors)
		save_dir = os.path.join(self.pointcloud_dir, uid)
		print("Saved pointcloud as " + str(save_dir), flush=True)
		os.makedirs(save_dir, exist_ok=True)
		filename_ply = os.path.join(save_dir, "pointcloud.ply")
		filename_npy = os.path.join(save_dir, "pointcloud.npz")

		if "ply" in self.args.save_file_type:
			pointcloud.export(filename_ply, file_type="ply")
		if "npz" in self.args.save_file_type:
			np.savez(filename_npy, points=points, normals=normals, colors=colors)

	def gen_interior_points(self, fragments, images, pixel_coords_in_camera, pixel_normals, uid):
		pix_to_face = fragments.pix_to_face
		V, H, W, F = pix_to_face.shape
		valid_v, valid_x, valid_y, first_valid_f = torch.where(pix_to_face[:, :, :, :1] != -1)
		last_pix_to_face = torch.cat([pix_to_face, -1 * torch.ones((V, H, W, 1), device=self.device)], dim=-1)
		last_valid_f = torch.argmin(last_pix_to_face, dim=-1)[valid_v, valid_x, valid_y] - 1

		first_pixel_coords = pixel_coords_in_camera[valid_v, valid_x, valid_y, first_valid_f] # (P, 3)
		last_pixel_coords = pixel_coords_in_camera[valid_v, valid_x, valid_y, last_valid_f] # (P, 3)


		P = first_pixel_coords.shape[0]
		random_indices = torch.randperm(P, device=self.device)[:self.args.num_interior_points]
		random_distances = torch.rand((self.args.num_interior_points, 1), device=self.device)

		points = first_pixel_coords[random_indices] * random_distances + last_pixel_coords[random_indices] * (1 - random_distances)

		points = points.cpu().numpy()

		pointcloud = trimesh.points.PointCloud(vertices=points)
		save_dir = os.path.join(self.interior_dir, uid)
		print("Saved interior pointcloud as " + str(save_dir), flush=True)
		os.makedirs(save_dir, exist_ok=True)
		filename_ply = os.path.join(save_dir, "interior.ply")
		filename_npy = os.path.join(save_dir, "interior.npz")

		if "ply" in self.args.save_file_type:
			pointcloud.export(filename_ply, file_type="ply")
		if "npz" in self.args.save_file_type:
			np.savez(filename_npy, points=points)	

	def forward(self, batch):
		# TODO: render by a batch
		for data in batch:
			mesh, uid, valid = data["mesh"], data["uid"], data["valid"]
			if not valid:
				# print(f"Mesh {uid} is not valid.", flush=True)
				continue
			print(f"Start render pointcloud of {uid}", flush=True)
			meshes = mesh.extend(self.num_views)
			fragments, images = self.render(meshes)
			if "png" in self.args.save_file_type:
				self.gen_image(images, uid)

			pixel_coords_in_camera, pixel_normals = self.get_pixel_data(meshes, fragments, images) 	# (N, P, K, 3)

			if self.args.get_render_points:	
				self.gen_pointcloud(fragments, images, pixel_coords_in_camera, pixel_normals, uid)
			if self.args.get_interior_points:
				self.gen_interior_points(fragments, images, pixel_coords_in_camera, pixel_normals, uid)
