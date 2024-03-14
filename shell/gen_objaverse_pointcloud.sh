export CUDA_VISIBLE_DEVICES=0
N_GPUS=1
objaverse_dir=data/Objaverse/objaverse
output_dir=data/Objaverse
log_dir=logs

torchrun \
--rdzv_endpoint localhost:26500 \
--nproc_per_node=${N_GPUS} \
main.py \
--objaverse_dir ${objaverse_dir} \
--output_dir ${output_dir} \
--log_dir ${log_dir} \
--num_workers 2 \
--total_uid_counts 40000 \
--num_points 200000 \
--image_size 624 \
--save_file_type "data" \
--resume \
####