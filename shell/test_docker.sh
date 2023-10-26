export CUDA_VISIBLE_DEVICES=0
N_GPUS=1
output_dir=/workspace/data/ShapeNet
log_dir='logs'

# torchrun \
# --rdzv_endpoint localhost:26500 \
# --nproc_per_node=${N_GPUS} \
python \
main.py \
--get_render_points \
--dataset "ShapeNet" \
--shapenet_mesh_dir /workspace/ShapeNetCore.v1 \
--shapenet_filelist_dir ${output_dir}/filelist \
--output_dir ${output_dir} \
--log_dir ${log_dir} \
--num_workers 0 \
--num_points 640000 \
--image_size 1024 \
--file_list "train_chair.txt" "test_chair.txt" \
--save_file_type "pointcloud" "data" \
# --resume \
# --debug \
# --camera_mode "Orthographic" \
######
