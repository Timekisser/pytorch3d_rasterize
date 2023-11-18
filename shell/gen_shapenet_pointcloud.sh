export CUDA_VISIBLE_DEVICES=0,1,2,3
N_GPUS=4
output_dir=data/ShapeNet
log_dir=logs

# torchrun \
# --rdzv_endpoint localhost:26500 \
# --nproc_per_node=${N_GPUS} \
python3 \
main.py \
--get_render_points \
--dataset "ShapeNet" \
--shapenet_mesh_dir ${output_dir}/ShapeNetCore.v1 \
--shapenet_filelist_dir ${output_dir}/filelist \
--output_dir ${output_dir} \
--output_folder "pointcloud_render_20w" \
--log_dir ${log_dir} \
--num_workers 4 \
--num_points 200000 \
--image_size 1024 \
--file_list "train_airplane.txt" "test_airplane.txt" "train_chair.txt" "test_chair.txt" "train_car.txt" "test_car.txt" "train_table.txt" "test_table.txt" "train_rifle.txt" "test_rifle.txt" \
--save_file_type "image" "data" "pointcloud" \
--camera_mode "Orthographic" \
--resume \
--save_memory
# --debug \
######
