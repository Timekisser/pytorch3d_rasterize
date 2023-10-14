export CUDA_VISIBLE_DEVICES=0,1,2,3
N_GPUS=4
output_dir='data/ShapeNet'
log_dir='logs'

torchrun \
--rdzv_endpoint localhost:26500 \
--nproc_per_node=${N_GPUS} \
main.py \
--get_render_points \
--dataset "ShapeNet" \
--shapenet_mesh_dir "data/ShapeNet/ShapeNetCore.v1" \
--shapenet_filelist_dir "data/ShapeNet/filelist" \
--output_dir "data/ShapeNet/" \
--log_dir ${log_dir} \
--num_workers 8 \
--num_points 640000 \
--image_size 1024 \
--file_list "train_chair.txt" "test_chair.txt" \
--save_file_type "pointcloud" "data" \
# --resume \
# --debug \
# --camera_mode "Orthographic" \
######
