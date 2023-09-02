export CUDA_VISIBLE_DEVICES=1,2
N_GPUS=2
output_dir='data/ShapeNet'
log_dir='logs'

torchrun \
--rdzv_endpoint localhost:26500 \
--nproc_per_node=${N_GPUS} \
main.py \
--resume \
--dataset "ShapeNet" \
--output_dir ${output_dir} \
--log_dir ${log_dir} \
--num_workers 8 \
--camera_mode "Orthographic" \
--get_interior_points \
--num_interior_points 100000 \
--faces_per_pixel 3 \
--save_file_type "ply" "npz" # "png" "glb", "obj"

