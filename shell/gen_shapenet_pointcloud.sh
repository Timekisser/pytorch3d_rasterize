export CUDA_VISIBLE_DEVICES=0
N_GPUS=1
output_dir=data/ShapeNet
log_dir='logs'

torchrun \
--rdzv_endpoint localhost:26500 \
--nproc_per_node=${N_GPUS} \
main.py \
--get_render_points \
--dataset "ShapeNet" \
--shapenet_mesh_dir ${output_dir}/ShapeNetCore.v1 \
--shapenet_filelist_dir ${output_dir}/filelist \
--output_dir ${output_dir} \
--log_dir ${log_dir} \
--num_workers 8 \
--num_points 640000 \
--image_size 1024 \
--file_list "train_chair.txt" "test_chair.txt" "train_airplane.txt" "test_airplane.txt" "train_car.txt" "test_car.txt" \
--save_file_type "pointcloud" "data" \
--resume \
# --debug \
# --camera_mode "Orthographic" \
######
