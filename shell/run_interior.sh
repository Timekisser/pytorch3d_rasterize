export CUDA_VISIBLE_DEVICES=6,7
N_GPUS=2
objaverse_dir='/mnt/sdc/weist/objaverse'
# output_dir='/mnt/sdb/xiongbj/Objaverse'
output_dir='data/Objaverse'
log_dir='logs'

torchrun \
--rdzv_endpoint localhost:26500 \
--nproc_per_node=${N_GPUS} \
main.py \
--resume \
--objaverse_dir ${objaverse_dir} \
--output_dir ${output_dir} \
--log_dir ${log_dir} \
--num_workers 8 \
--total_uid_counts 8000000 \
--num_points 500000 \
--get_interior_points \
--num_interior_points 50000 \
--faces_per_pixel 6 \
--save_file_type "ply" "png" "npz" #, "glb", "obj"

