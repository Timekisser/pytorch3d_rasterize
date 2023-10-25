export CUDA_VISIBLE_DEVICES=0
N_GPUS=1
output_dir=data/ShapeNet
log_dir='logs'

# torchrun \
# --rdzv_endpoint localhost:26500 \
# --nproc_per_node=${N_GPUS} \
python \
main.py \
--get_render_points \
--dataset "ShapeNet" \
--shapenet_mesh_dir ${output_dir}/ShapeNetCore.v1 \
--shapenet_filelist_dir ${output_dir}/filelist \
--output_dir ${output_dir} \
--log_dir ${log_dir} \
--num_workers 8 \
--num_points 100000 \
--image_size 600 \
--file_list "train_chair.txt" "test_chair.txt" \
--save_file_type "pointcloud" "data" \
--camera_mode "Orthographic" \
# --resume \
# --debug \
######
