#--------------------------------------------------------
# Dual Octree Graph Networks
import torch
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data.sampler import BatchSampler, SequentialSampler
import argparse
import os
import shutil
import sys
import multiprocessing as mp
from tqdm import tqdm
from dataset.objaverse import ObjaverseDataset
from dataset.shapenet import ShapeNetDataset, ShapeNetFileList
from dataset.prefetch import DataPreFetcher 
from models.render import PointCloudRender
from utils.distributed import (
	get_rank,
	synchronize,
	get_world_size,
)
os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def build_dataloader(args):
	if args.dataset == "Objaverse":
		dataset = ObjaverseDataset(args)
	elif args.dataset == "ShapeNet":
		dataset = ShapeNetDataset(args)

	if args.distributed:
		sampler = DistributedSampler(dataset, shuffle=False)
	else:
		sampler = SequentialSampler(dataset)
	batch_sampler = BatchSampler(sampler, batch_size=1, drop_last=False)
	data_loader = DataLoader(
		dataset=dataset,
		num_workers=args.num_workers,
		batch_sampler=batch_sampler,
		collate_fn=dataset.collect_fn,
	)
	return data_loader

def mesh_to_cuda(batch,device):
	for data in batch:
		if data['mesh'] is not None:
			data['mesh'] = data['mesh'].to(device)

def generate_pointcloud(args):
	model = PointCloudRender(
		args=args,
		image_size=args.image_size,
		output_dir=args.output_dir,
		device=args.device,
	)
	model.to(args.device)
	data_loader = build_dataloader(args)

	# fetcher = DataPreFetcher(data_loader, args.device)
	# batch = fetcher.next()
	# for i in range(len(data_loader)):
	for batch in tqdm(data_loader):
		mesh_to_cuda(batch, model.device)
		with torch.no_grad():
			model(batch)

		# torch.cuda.empty_cache()
		# batch = fetcher.next()

def shapenet_mesh_repair(args, num_processes=4):
	gpu_ids = os.environ['CUDA_VISIBLE_DEVICES'].split(",")
	world_size = len(gpu_ids)
	print(num_processes, gpu_ids, flush=True)
	filelist = ShapeNetFileList(args, args.total_uid_counts)
	mesh_dir = args.shapenet_mesh_dir
	num_meshes = len(filelist.uids)
	mesh_per_process = num_meshes // num_processes

	def process(process_id):
		os.environ['CUDA_VISIBLE_DEVICES']=str(gpu_ids[process_id % world_size])
		for i in tqdm(range(process_id * mesh_per_process, (process_id + 1)* mesh_per_process), ncols=80):
			if i >= num_meshes:
				continue
			uid = filelist.uids[i]
			print(f"Repairing mesh {uid}", flush=True)
			folder_obj = os.path.join(mesh_dir, uid)
			folder_repair = os.path.join(args.output_dir, "mesh_repair", uid)
			filename_obj = os.path.join(folder_obj, "model.obj")
			filename_repair = os.path.join(folder_repair, "model.obj")
			
			os.makedirs(folder_repair, exist_ok=True)
			shutil.copytree(folder_obj, folder_repair, dirs_exist_ok=True)
			os.rename(filename_repair, os.path.join(folder_repair, "origin.obj"))
			command = f"./utils/RayCastMeshRepair --input {filename_obj} --output {filename_repair}"
			output = os.system(command)
			assert output == 0
	
	if num_processes == 1:
		process(0)
	else:
		processes = [mp.Process(target=process, args=[pid]) for pid in range(num_processes)]
		for p in processes:
			p.start()
		for p in processes:
			p.join()

if __name__ == "__main__":
	# torch.multiprocessing.set_start_method('spawn')
	parser = argparse.ArgumentParser("Generate Pointcloud")

	# DDP settings
	parser.add_argument("--device", default="cuda", type=str)
	parser.add_argument("--batch_size", default=1, type=int)
	parser.add_argument("--backend", type=str, default="gloo", help="which backend to use")
	parser.add_argument("--num_workers", default=0, type=int)

	# Dataset settings
	parser.add_argument("--dataset", default='Objaverse', type=str)
	parser.add_argument("--resume", action="store_true")
	parser.add_argument("--debug", action="store_true")
	parser.add_argument("--total_uid_counts", default=8000000, type=int)
	parser.add_argument("--output_dir", default='data/Objaverse', type=str)
	parser.add_argument("--pointcloud_folder", default='pointcloud', type=str)
	parser.add_argument("--image_folder", default='image', type=str)
	# Objaverse
	parser.add_argument("--have_category", action="store_true")
	parser.add_argument("--objaverse_dir", default="/mnt/sdc/weist/objaverse", type=str)

	# ShapeNet
	parser.add_argument("--shapenet_mesh_dir", default="data/ShapeNet/ShapeNetCore.v1/", type=str)
	parser.add_argument("--shapenet_filelist_dir", default="data/ShapeNet/filelist", type=str)
	parser.add_argument("--file_list", default=["train_airplane.txt", "test_airplane.txt"], type=str, nargs="+")
	parser.add_argument("--log_dir", default='logs', type=str)
	parser.add_argument("--save_file_type", default=["pointcloud", "image", "data", "normal", "origin", "object"], type=str, nargs="+")
	
	# Render settings
	parser.add_argument("--camera_mode", default="Perspective", type=str)
	parser.add_argument("--bin_mode", default="coarse", choices=["coarse", "naive"], type=str, help="Naive mode do not get warnings but is slower.")
	parser.add_argument("--num_points", default=500000, type=int)	
	parser.add_argument("--num_interior_points", default=50000, type=int)	
	parser.add_argument("--image_size", default=600, type=int)	
	parser.add_argument("--points_dilate", default=0.005, type=float)
	parser.add_argument("--faces_per_pixel", default=1, type=int)
	parser.add_argument("--get_interior_points", action="store_true")
	parser.add_argument("--get_render_points", action="store_true")
	parser.add_argument("--mesh_repair", action="store_true")
	parser.add_argument("--cull_backfaces", action="store_true")
	parser.add_argument("--save_memory", action="store_true")

	args = parser.parse_args()

	os.makedirs(args.log_dir, exist_ok=True)
	sys.stdout = open(os.path.join(args.log_dir, "stdout.txt"), "w")
	# sys.stderr = open(os.path.join(args.log_dir, "stderr.txt"), "w")
	
	# init_distributed_mode(args)
	n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
	args.distributed = n_gpu > 1
	args.local_rank = int(os.environ["LOCAL_RANK"]) if "LOCAL_RANK" in os.environ else 0
	if args.distributed:
		torch.cuda.set_device(args.local_rank)
		torch.distributed.init_process_group(backend=args.backend, init_method="env://")
		synchronize()

	print(args, flush=True)
	if args.mesh_repair:
		if args.dataset == "ShapeNet":
			shapenet_mesh_repair(args)
	else:
		generate_pointcloud(args)
	
	sys.stdout.close()
	sys.stderr.close()
