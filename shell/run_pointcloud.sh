export CUDA_VISIBLE_DEVICES=4,5,6,7
N_GPUS=4
objaverse_dir='/mnt/sdc/weist/objaverse'
# output_dir='/mnt/sdb/xiongbj/Objaverse'
output_dir='data/Objaverse'
log_dir='logs'

torchrun \
--rdzv_endpoint localhost:26500 \
--nproc_per_node=${N_GPUS} \
main.py \
--resume \
--get_render_points \
--objaverse_dir ${objaverse_dir} \
--output_dir ${output_dir} \
--log_dir ${log_dir} \
--num_workers 8 \
--total_uid_counts 8000000 \
--num_points 500000 \
--save_file_type "ply" "npz" \
--debug
