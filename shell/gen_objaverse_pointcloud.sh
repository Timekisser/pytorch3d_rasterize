export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
N_GPUS=8
objaverse_dir=data/Objaverse/objaverse
output_dir=data/Objaverse
log_dir=logs

torchrun \
--rdzv_endpoint localhost:26500 \
--nproc_per_node=${N_GPUS} \
--max-restarts=100 \
--rdzv-backend=c10d \
main.py \
--dataset "Objaverse" \
--objaverse_dir ${objaverse_dir} \
--output_dir ${output_dir} \
--pointcloud_folder "pointcloud_20w" \
--log_dir ${log_dir} \
--num_workers 4 \
--total_uid_counts 8000000 \
--num_points 200000 \
--image_size 624 \
--save_file_type "data" \
--resume \
####