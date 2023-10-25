import torch
import torch.nn
import numpy as np
import os
import trimesh
import traceback
import matplotlib.pyplot as plt
from pyrender import Mesh, Scene, Viewer, Node
from pyrender import OrthographicCamera, PerspectiveCamera
from pyrender import OffscreenRenderer, RenderFlags
from scipy.spatial.transform import Rotation

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

		# self.elevation = self.elevation * self.batch_size
		# self.azim_angle = self.azim_angle * self.batch_size

		self.cameras = self.get_cameras()
		self.renderer = OffscreenRenderer(viewport_width=image_size, viewport_height=image_size, point_size=1.0)
		# self.shader = self.get_shader()
		self.full_transform = None
		# Output dir
		self.image_dir = os.path.join(output_dir, "image")
		self.pointcloud_dir = os.path.join(output_dir, "pointcloud_test")
		for dir in [self.image_dir, self.pointcloud_dir]:
			os.makedirs(dir, exist_ok=True)	
	
	def get_cameras(self):
	   	# Initialize the camera with camera distance, elevation, azimuth angle,
		# and image size
		
		if self.args.camera_mode == "Orthographic":
			cameras = []
			for elev, azim in zip(self.elevation, self.azim_angle):
				R = np.diag([0, 0, 0, 1.0])
				T = np.diag([1, 1, 1, 1.0])
				r = Rotation.from_euler('xyz', [elev / 180.0 * np.pi, azim / 180.0 * np.pi, 0.0])
				R[:3, :3] = r.as_matrix()
				T[2, 3] = self.camera_dist

				cam = OrthographicCamera(xmag=1.2, ymag=1.2, znear=0.01, zfar=100.0)
				nc = Node(camera=cam, matrix=np.dot(R, T))
				cameras.append(nc)
		else:
			raise Exception("No such camera mode.")
		return cameras
	
	def render(self, mesh, uid):
		pyrender_mesh = Mesh.from_trimesh(mesh)
		node_mesh = Node(mesh=pyrender_mesh, matrix=np.diag([1, 1, 1, 1]))

		color_list, position_list, normal_list = [], [], []
		for cam_id, node_camera in enumerate(self.cameras):
			scene = Scene(ambient_light=[1.0, 1.0, 1.0], bg_color=[-1.0, -1.0, -1.0, -1.0])
			scene.add_node(node_camera)
			scene.add_node(node_mesh)
			color, normal = self.renderer.render(scene, flags=RenderFlags.FLAT | RenderFlags.FACE_NORMALS | RenderFlags.CLEAR | RenderFlags.SKIP_CULL_FACES)
			position = self.renderer.render(scene, flags=RenderFlags.FRAGMENT | RenderFlags.NORMAL_ONLY | RenderFlags.SKIP_CULL_FACES)[0]
			
			if "image" in self.args.save_file_type:
				save_dir = os.path.join(self.image_dir, uid)
				os.makedirs(save_dir, exist_ok=True)
				filename_png = os.path.join(save_dir, f"image_{cam_id}.png")
				plt.cla()
				plt.imshow(color)
				plt.savefig(filename_png)

			valid = np.logical_and(position[..., -1] > 0, color[..., -1] > 0)
			color = color[valid]
			position = position[valid]
			normal = normal[valid]

			color_list.append(color)
			position_list.append(position)
			normal_list.append(normal)
		
		colors = np.concatenate(color_list, axis=0)
		positions = np.concatenate(position_list, axis=0)
		normals = np.concatenate(normal_list, axis=0)

		P = colors.shape[0]
		random_indices = np.random.randint(0, P, size=(self.num_points, ))
		points = positions[random_indices][..., :3]
		normals = normals[random_indices][..., :3]
		colors = colors[random_indices]
		
		pointcloud = trimesh.points.PointCloud(vertices=points[..., :3], colors=(colors * 255.).astype(np.uint8))
		save_dir = os.path.join(self.pointcloud_dir, uid)
		os.makedirs(save_dir, exist_ok=True)
		filename_ply = os.path.join(save_dir, "pointcloud.ply")
		filename_npy = os.path.join(save_dir, "pointcloud.npz")

		if "pointcloud" in self.args.save_file_type:
			pointcloud.export(filename_ply, file_type="ply")
		if "data" in self.args.save_file_type:
			np.savez(filename_npy, points=points, normals=normals, colors=colors[..., :3])
		if "normal" in self.args.save_file_type:
			self.visualize_points_and_normals(points, normals, uid)



	def gen_pointcloud(self, fragments, meshes, pixel_coords_in_camera, pixel_normals, uid):
		valid_v, valid_x, valid_y = torch.where(fragments.pix_to_face[:, :, :, 0] != -1)
		if valid_v.shape[0] == 0:
			raise Exception("Empty mesh.")
		pixel_coords = pixel_coords_in_camera[valid_v, valid_x, valid_y, 0] # (P, 3)
		pixel_normals = pixel_normals[valid_v, valid_x, valid_y, 0]	# (P, 3)
		texels = meshes.sample_textures(fragments).squeeze(-2)
		pixel_colors = texels[valid_v, valid_x, valid_y, :]	# (P, 3)

		# 正常来说得到的法向量norm都应该是1
		# 但有部分为0，有部分在0~1
		normals_norm = torch.norm(pixel_normals, p=2, dim=1)
		normals_valid = normals_norm > 0.9
		pixel_coords = pixel_coords[normals_valid]
		pixel_normals = pixel_normals[normals_valid]
		pixel_colors = pixel_colors[normals_valid]

		P = pixel_coords.shape[0]
		random_indices = torch.randint(0, P, size=(self.num_points, ), device=self.device)
		points = pixel_coords[random_indices]
		normals = pixel_normals[random_indices]
		normals = normals / torch.norm(normals, p=2, dim=1, keepdim=True)
		colors = pixel_colors[random_indices]

		# normal directions
		camera_centers = self.cameras.get_camera_center()
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
		# print("Saved pointcloud as " + str(save_dir), flush=True)
		os.makedirs(save_dir, exist_ok=True)
		filename_ply = os.path.join(save_dir, "pointcloud.ply")
		filename_npy = os.path.join(save_dir, "pointcloud.npz")

		if "pointcloud" in self.args.save_file_type:
			pointcloud.export(filename_ply, file_type="ply")
		if "data" in self.args.save_file_type:
			np.savez(filename_npy, points=points, normals=normals, colors=colors)
		if "normal" in self.args.save_file_type:
			self.visualize_points_and_normals(points, normals, uid)
	
	def visualize_points_and_normals(self, points, normals, uid):
		points_with_normals = np.concatenate([points + 0.01 * normals, points], axis=0)
		colors = np.concatenate([np.array([[1, 0, 0]] * points.shape[0]), np.array([[0, 1, 0]] * points.shape[0])], axis=0) * 255.0
		pointcloud = trimesh.points.PointCloud(vertices=points_with_normals, colors=colors)
		save_dir = os.path.join(self.pointcloud_dir, uid, "normals.ply")
		os.makedirs(os.path.dirname(save_dir), exist_ok=True)
		pointcloud.export(save_dir, file_type="ply")


	def forward(self, batch):
		# TODO: render by a batch
		for data in batch:
			mesh, uid, valid = data["mesh"], data["uid"], data["valid"]
			if not valid:
				continue
			# try:
			self.render(mesh, uid)
			# except:
			# 	print(f"Render Error in mesh {uid}.", flush=True)
			# 	print(traceback.format_exc(), flush=True)
			# 	if self.args.debug:
			# 		raise ValueError
