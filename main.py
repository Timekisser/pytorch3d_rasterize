# --------------------------------------------------------
# Dual Octree Graph Networks
import torch
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data.sampler import BatchSampler, SequentialSampler
import argparse
import os
from dataset.mesh import MeshDataset, DataPreFetcher
from models.render import PointCloudRender
from utils.distributed_utils import init_distributed_mode

def build_dataloader(args):
	dataset = MeshDataset(args, device=args.device)
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


def generate_pointcloud(args):
	model = PointCloudRender(
		args=args, 
		batch_size=args.batch_size,
		device=args.device
	)
	model.to(args.device)
	data_loader = build_dataloader(args)

	# fetcher = DataPreFetcher(data_loader, args.device)
	# mesh, uid = fetcher.next()
	# for i in range(len(data_loader)):
	for batch in data_loader:
		model(batch)
		# mesh, uid = fetcher.next()

if __name__ == "__main__":
	parser = argparse.ArgumentParser("Objaverse Pointcloud")
	parser.add_argument("--device", default="cuda", type=str)
	parser.add_argument("--num_gpus", default=1, type=int)
	parser.add_argument("--batch_size", default=1, type=int)
	parser.add_argument("--num_workers", default=0, type=int)	
	parser.add_argument("--total_uid_counts", default=1, type=int)
	parser.add_argument("--output_dir", default='data/Objaverse', type=str)
	parser.add_argument("--camera_mode", default="Perspective", type=str)
	parser.add_argument("--objaverse_dir", default="/mnt/sdc/weist/objaverse", type=str)
	parser.add_argument("--save_file_type", default=["ply", "png", "npz", "glb", "obj"], type=list)
	args = parser.parse_args()
	init_distributed_mode(args)
	generate_pointcloud(args)