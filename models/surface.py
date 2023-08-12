# def check_folder(filenames: list):
# 	r''' Checks whether the folder contains the filename exists.
# 	'''
# 	for filename in filenames:
# 		folder = os.path.dirname(filename)
# 		if not os.path.exists(folder):
# 			os.makedirs(folder)

# def run_mesh2sdf():
# 	r''' Converts the meshes from ShapeNet to SDFs and manifold meshes.
# 	'''

# 	print('-> Run mesh2sdf.')
# 	mesh_scale = 0.8
# 	for filename, filename_raw in glbs.items():
# 		filename_obj = os.path.join(root_folder, 'mesh', filename + '.obj')
# 		filename_box = os.path.join(root_folder, 'bbox', filename + '.npz')
# 		filename_npy = os.path.join(root_folder, 'sdf', filename + '.npy')
# 		check_folder([filename_obj, filename_box, filename_npy])

# 		# load the raw mesh
# 		mesh = trimesh.load(filename_raw, force='mesh')

# 		# rescale mesh to [-1, 1] for mesh2sdf, note the factor **mesh_scale**
# 		vertices = mesh.vertices
# 		bbmin, bbmax = vertices.min(0), vertices.max(0)
# 		center = (bbmin + bbmax) * 0.5
# 		scale = 2.0 * mesh_scale / (bbmax - bbmin).max()
# 		vertices = (vertices - center) * scale

# 		# run mesh2sdf
# 		sdf, mesh_new = mesh2sdf.compute(vertices, mesh.faces, size, fix=True,
# 										level=level, return_mesh=True)
# 		mesh_new.vertices = mesh_new.vertices * shape_scale

# 		# save
# 		np.savez(filename_box, bbmax=bbmax, bbmin=bbmin, mul=mesh_scale)
# 		np.save(filename_npy, sdf)
# 		mesh_new.export(filename_obj)

# def run_mesh2sdf_mp():
# 	r''' Converts the meshes from ShapeNet to SDFs and manifold meshes.
# 		'''

# 	num_processes = mp.cpu_count()
# 	num_meshes = len(glbs)
# 	mesh_per_process = num_meshes // num_processes + 1

# 	print('-> Run mesh2sdf.')
# 	mesh_scale = 0.8
# 	filenames = list(glbs.keys())
# 	filepaths = list(glbs.values())
# 	def process(process_id):
# 		for i in tqdm(range(process_id * mesh_per_process, (process_id + 1)* mesh_per_process), ncols=80):
# 			if i >= num_meshes: break
# 			filename = filenames[i]
# 			filename_raw = filepaths[i]
# 			filename_obj = os.path.join(root_folder, 'mesh', filename + '.obj')
# 			filename_box = os.path.join(root_folder, 'bbox', filename + '.npz')
# 			filename_npy = os.path.join(root_folder, 'sdf', filename + '.npy')
# 			check_folder([filename_obj, filename_box, filename_npy])
# 			if os.path.exists(filename_obj): continue

# 			# load the raw mesh
# 			mesh = trimesh.load(filename_raw, force='mesh')

# 			# rescale mesh to [-1, 1] for mesh2sdf, note the factor **mesh_scale**
# 			vertices = mesh.vertices
# 			bbmin, bbmax = vertices.min(0), vertices.max(0)
# 			center = (bbmin + bbmax) * 0.5
# 			scale = 2.0 * mesh_scale / (bbmax - bbmin).max()
# 			vertices = (vertices - center) * scale

# 			# run mesh2sdf
# 			sdf, mesh_new = mesh2sdf.compute(vertices, mesh.faces, size, fix=True,
# 											level=level, return_mesh=True)
# 			mesh_new.vertices = mesh_new.vertices * shape_scale

# 			# save
# 			np.savez(filename_box, bbmax=bbmax, bbmin=bbmin, mul=mesh_scale)
# 			np.save(filename_npy, sdf)
# 			mesh_new.export(filename_obj)

# 	processes = [mp.Process(target=process, args=[pid]) for pid in range(num_processes)]
# 	for p in processes:
# 		p.start()
# 	for p in processes:
# 		p.join()

# def barycentric_interpolation(points, values, interp_points):

# 	# 计算重心坐标
# 	v0 = interp_points - points[:, 0, :]
# 	v1 = interp_points - points[:, 1, :]
# 	v2 = interp_points - points[:, 2, :] 

# 	s0 = np.linalg.norm(np.cross(v1, v2, axis=1), axis=1)
# 	s1 = np.linalg.norm(np.cross(v2, v0, axis=1), axis=1)
# 	s2 = np.linalg.norm(np.cross(v0, v1, axis=1), axis=1)
	
# 	S = s0 + s1 + s2
# 	w0 = (s0 / S)[:, None]
# 	w1 = (s1 / S)[:, None]
# 	w2 = (s2 / S)[:, None]

# 	# 计算插值值
# 	interp_values = w0 * values[:, 0, :] + w1 * values[:, 1, :] + w2 * values[:, 2, :]

# 	return interp_values 

# def gen_pts_by_surface(geometry, num_samples, filename_ply):
# 	points, face_idx = trimesh.sample.sample_surface(geometry, num_samples)
# 	normals = geometry.face_normals[face_idx]

# 	visual = geometry.visual
# 	print(visual.kind)
# 	material = visual.material
# 	if material.baseColorTexture != None:
# 		vertex_colors = trimesh.visual.color.uv_to_color(visual.uv, visual.material.baseColorTexture)    
# 		face_to_vertex = geometry.faces[face_idx]             # (N, 3)
# 		face_vertex_pos = geometry.vertices[face_to_vertex]   # (N, 3, 3)
# 		face_vertex_color = vertex_colors[face_to_vertex]
# 		colors = barycentric_interpolation(face_vertex_pos, face_vertex_color, points)
# 	elif material.baseColorFactor is not None:
# 		colors = np.tile(material.baseColorFactor, (num_samples, 1))
# 	else:
# 		colors = np.tile(material.main_color, (num_samples, 1))
	
# 	bbox_min = np.min(points, axis=0)
# 	bbox_max = np.max(points, axis=0)
# 	bbox_mid = (bbox_max + bbox_min) / 2
# 	points = (points - bbox_mid) / (bbox_max - bbox_min) * 2 # [-1.0, 1.0]
# 	points = points * shape_scale
# 	pointcloud = o3d.geometry.PointCloud()
# 	pointcloud.points = o3d.utility.Vector3dVector(points)
# 	pointcloud.colors = o3d.utility.Vector3dVector(colors[:, :3] / 255.0)
# 	o3d.io.write_point_cloud(filename_ply, pointcloud)