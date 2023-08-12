# --------------------------------------------------------
# Dual Octree Graph Networks
import torch
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data.sampler import BatchSampler, SequentialSampler
import argparse

from dataset.mesh import MeshDataset, DataPreFetcher
from models.render import PointCloudRender
from utils.distributed_utils import init_distributed_mode

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


def generate_pointcloud(args):
	model = PointCloudRender(device=args.device)
	model.to(args.device)
	data_loader = build_dataloader(args)

	fetcher = DataPreFetcher(data_loader, args.device)
	# mesh, uid = fetcher.next()
	# for i in range(len(data_loader)):
	for mesh, uid in data_loader:
		print(f"Start render pointcloud of {uid}")
		model(mesh, uid)
		# mesh, uid = fetcher.next()

if __name__ == '__main__':
	parser = argparse.ArgumentParser("Objaverse Pointcloud")
	parser.add_argument("--device", default="cpu", type=str)
	parser.add_argument("--num_gpus", default=1, type=int)
	parser.add_argument('--num_workers', default=0, type=int)	
	parser.add_argument('--total_uid_counts', default=1, type=int)
	parser.add_argument('--output_dir', default='data/Objaverse', type=str)
	args = parser.parse_args()
	init_distributed_mode(args)
	generate_pointcloud(args)