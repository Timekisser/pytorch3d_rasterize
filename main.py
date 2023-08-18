#--------------------------------------------------------
# Dual Octree Graph Networks
import torch
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data.sampler import BatchSampler, SequentialSampler
import argparse
import os
import sys
from dataset.mesh import MeshDataset, DataPreFetcher
from models.render import PointCloudRender
from utils.distributed import (
    get_rank,
    synchronize,
)

def build_dataloader(args):
	dataset = MeshDataset(args)
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
		output_dir = args.output_dir,
		device=args.device
	)
	model.to(args.device)
	data_loader = build_dataloader(args)

	# fetcher = DataPreFetcher(data_loader, args.device)
	# batch = fetcher.next()
	# for i in range(len(data_loader)):
	for batch in data_loader:
		mesh_to_cuda(batch, model.device)
		model(batch)
		# batch = fetcher.next()

if __name__ == "__main__":
	torch.multiprocessing.set_start_method('spawn')
	parser = argparse.ArgumentParser("Objaverse Pointcloud")

	# DDP settings
	parser.add_argument("--device", default="cuda", type=str)
	parser.add_argument("--batch_size", default=1, type=int)
	parser.add_argument("--backend", type=str, default="gloo", help="which backend to use")
	parser.add_argument("--num_workers", default=0, type=int)

	# Dataset settings
	parser.add_argument("--resume", default=True, type=str, help="Continue processing.")
	parser.add_argument("--total_uid_counts", default=8000000, type=int)
	parser.add_argument("--output_dir", default='data/Objaverse', type=str)
	parser.add_argument("--objaverse_dir", default="/mnt/sdc/weist/objaverse", type=str)
	parser.add_argument("--log_dir", default='logs', type=str)
	parser.add_argument("--save_file_type", default=["ply", "png", "npz", "glb", "obj"], type=str, nargs="+")
	
	# Render settings
	parser.add_argument("--camera_mode", default="Perspective", type=str)
	parser.add_argument("--bin_mode", default="coarse", choices=["coarse", "naive"], type=str, help="Naive mode do not get warnings but is slower.")
	parser.add_argument("--num_points", default=500000, type=int)	
	args = parser.parse_args()

	sys.stdout = open(os.path.join(args.log_dir, "stdout.txt"), "w")
	sys.stderr = open(os.path.join(args.log_dir, "stderr.txt"), "w")
	
	# init_distributed_mode(args)
	n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
	args.distributed = n_gpu > 1
	args.local_rank = int(os.environ["LOCAL_RANK"]) if "LOCAL_RANK" in os.environ else 0
	if args.distributed:
		torch.cuda.set_device(args.local_rank)
		torch.distributed.init_process_group(backend=args.backend, init_method="env://")
		synchronize()

	print(args, flush=True)
	generate_pointcloud(args)
	
	sys.stdout.close()
	sys.stderr.close()
