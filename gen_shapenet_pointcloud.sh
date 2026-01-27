export CUDA_VISIBLE_DEVICES=4,5,6,7
N_GPUS=4
output_dir=data/ShapeNet
input_dir=/mnt/sdc/weist/data/ShapeNet/
log_dir=logs

# torchrun \
# --rdzv_endpoint localhost:26500 \
# --nproc_per_node=${N_GPUS} \

sudo /home/wangjh/anaconda3/envs/pt3d/bin/python \
main.py \
--dataset "ShapeNet" \
--shapenet_mesh_dir ${input_dir}/ShapeNetCore.v1 \
--shapenet_filelist_dir ${input_dir}/filelist \
--output_dir ${output_dir} \
--pointcloud_folder "pointcloud_render_20w" \
--image_folder "image_1024" \
--log_dir ${log_dir} \
--num_workers 4 \
--num_points 200000 \
--image_size 1024 \
--file_list "train_im_5.txt" "test_im_5.txt" \
--save_file_type "image" \
--camera_mode "Perspective" \
--resume \
# --camera_mode "Orthographic" \
# --save_memory
# --debug \
# "train_chair.txt" "test_chair.txt" "train_car.txt" "test_car.txt" "train_table.txt" "test_table.txt" "train_rifle.txt" "test_rifle.txt" 
# --debug \
######
