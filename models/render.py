import torch
import torch.nn
import os
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
	def __init__(self, image_size=1024, camera_dist=3, enable_image=True, device="cuda") -> None:
		super().__init__()
		self.image_size = image_size
		self.camera_dist = camera_dist
		self.elevation =  [0, 0,  0,   0,   -90, 90]
		self.azim_angle = [0, 90, 180, 270, 0,   0]
		self.batch_size = len(self.elevation)
		self.num_points = 100000
		self.device = device

		self.renderer = self.get_renderer()

		# Output dir
		base_dir = "data/Objaverse/"
		self.image_dir = os.path.join(base_dir, "image")
		self.pointcloud_dir = os.path.join(base_dir, "pointcloud") 
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
		for i in range(self.batch_size):
			elev = self.elevation[i]
			azim = self.azim_angle[i]
			filename_png = os.path.join(self.image_dir, f"{uid}_elev{int(elev)}_azim{int(azim)}.png")
			plt.imshow(images[i, ..., :3].cpu().numpy())
			plt.savefig(filename_png)
			plt.cla()
			print("Saved image as " + str(filename_png))

	def gen_pointcloud(self, meshes, fragments, uid):
		verts = meshes.verts_packed()  # (N, V, 3)
		faces = meshes.faces_packed()  # (N, F, 3)
		texels = meshes.sample_textures(fragments)
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
		pixel_texels = texels[:, valid_x, valid_y, 0]	# (N, P, 3)
		
		N, P = pixel_coords.shape[0], pixel_coords.shape[1]
		random_index = torch.randint(0, N * P, (self.num_points, ))
		points = pixel_coords.reshape(1, -1, 3)[:, random_index]
		normals = pixel_normals.reshape(1, -1, 3)[:, random_index]
		texels = pixel_texels.reshape(1, -1, 3)[:, random_index]

		pointcloud = Pointclouds(points=points, normals=normals, features=texels)
		filename = os.path.join(self.pointcloud_dir, f"{uid}.ply")
		IO().save_pointcloud(pointcloud, filename)

	def forward(self, mesh, uid):
		meshes = mesh.extend(self.batch_size)
		fragments, images = self.render(meshes)

		if self.enable_image:
			self.gen_image(images, uid)
		self.gen_pointcloud(meshes, fragments, uid)