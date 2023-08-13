import torch
import torch.nn
import numpy as np
import os
import trimesh
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

class PointCloudRender(torch.nn.Module):
	def __init__(self, args, image_size=1024, camera_dist=3, batch_size=1, enable_image=True, output_dir="data/Objaverse",  device="cuda") -> None:
		super().__init__()
		self.args = args
		self.image_size = image_size
		self.camera_dist = camera_dist
		self.elevation =  [0, 0,  0,   0,   -90, 90]
		self.azim_angle = [0, 90, 180, 270, 0,   0]
		self.batch_size = batch_size
		self.num_views = len(self.elevation)
		self.num_points = 100000
		self.device = device

		self.renderer = self.get_renderer()

		# Output dir
		self.image_dir = os.path.join(output_dir, "image")
		self.pointcloud_dir = os.path.join(output_dir, "pointcloud") 
		os.makedirs(self.image_dir, exist_ok=True)
		os.makedirs(self.pointcloud_dir, exist_ok=True)
		self.enable_image = enable_image

	def get_renderer(self):
	   	# Initialize the camera with camera distance, elevation, azimuth angle,
		# and image size
		R, T = look_at_view_transform(dist=self.camera_dist, elev=self.elevation, azim=self.azim_angle, device=self.device)
		# cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
		cameras = FoVOrthographicCameras(R=R, T = T, device=self.device)
		raster_settings = RasterizationSettings(
			image_size=self.image_size,
			blur_radius=0.0,
			faces_per_pixel=1,
		)
		# Initialize rasterizer by using a MeshRasterizer class
		rasterizer = MeshRasterizer(
			cameras=cameras,
			raster_settings=raster_settings
		)
		# The textured phong shader interpolates the texture uv coordinates for
		# each vertex, and samples from a texture image.
		# lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])
		shader = SoftPhongShader(cameras=cameras, device=self.device)
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
		print("Saved image as " + str(save_dir))
		for i in range(self.batch_size):
			elev = self.elevation[i]
			azim = self.azim_angle[i]
			filename_png = os.path.join(save_dir, f"elev{int(elev)}_azim{int(azim)}.png")
			plt.imshow(images[i, ..., :3].cpu().numpy())
			plt.savefig(filename_png)
			plt.cla()
			
	def gen_pointcloud(self, meshes, fragments, images, uid):
		verts = meshes.verts_packed()  # (N, V, 3)
		faces = meshes.faces_packed()  # (N, F, 3)
		# texels = meshes.sample_textures(fragments)
		vertex_normals = meshes.verts_normals_packed()  # (N, V, 3)
		faces_verts = verts[faces]
		faces_normals = vertex_normals[faces]
		valid_x, valid_y = torch.where(fragments.zbuf != -1)[1:3]
		pixel_coords_in_camera = interpolate_face_attributes(
			fragments.pix_to_face, fragments.bary_coords, faces_verts
		)
		pixel_normals = interpolate_face_attributes(
			fragments.pix_to_face, fragments.bary_coords, faces_normals
		)
		pixel_coords = pixel_coords_in_camera[:, valid_x, valid_y, 0] # (N, P, 3)
		pixel_normals = pixel_normals[:, valid_x, valid_y, 0]	# (N, P, 3)
		pixel_colors = images[:, valid_x, valid_y, :]	# (N, P, 4)
		
		N, P = pixel_coords.shape[0], pixel_coords.shape[1]
		random_indices = torch.randint(0, N * P, (self.num_points, ), device=self.device)
		rows = random_indices // P
		cols = random_indices % P
		points = pixel_coords[rows, cols]
		normals = pixel_normals[rows, cols]
		colors = pixel_colors[rows, cols]

		points = points.cpu().numpy()
		normals = normals.cpu().numpy()
		colors = colors.cpu().numpy()

		pointcloud = trimesh.points.PointCloud(vertices=points, colors=colors)
		save_dir = os.path.join(self.pointcloud_dir, uid)
		print("Saved pointcloud as " + str(save_dir))
		os.makedirs(save_dir, exist_ok=True)
		filename_xyz = os.path.join(save_dir, "pointcloud.ply")
		filename_npy = os.path.join(save_dir, "pointcloud.npz") 
		pointcloud.export(filename_xyz, file_type="ply")
		np.savez(filename_npy, points=points, normals=normals, colors=colors)


	def forward(self, batch_mesh, batch_uid):
		# TODO: render by a batch
		for mesh, uid in zip(batch_mesh, batch_uid):
			meshes = mesh.extend(self.num_views)
			fragments, images = self.render(meshes)
			if self.enable_image:
				self.gen_image(images, uid)
			self.gen_pointcloud(meshes, fragments, images, uid)