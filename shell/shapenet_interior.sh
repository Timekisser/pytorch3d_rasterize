export CUDA_VISIBLE_DEVICES=6,7
N_GPUS=2
output_dir='data/ShapeNet'
log_dir='logs'

torchrun \
--rdzv_endpoint localhost:26500 \
--nproc_per_node=${N_GPUS} \
main.py \
--resume \
--output_dir ${output_dir} \
--log_dir ${log_dir} \
--num_workers 8 \
--num_points 100000 \
--get_interior_points \
--num_interior_points 50000 \
--faces_per_pixel 5 \
--save_file_type "ply" "npz" # "png" "glb", "obj"

