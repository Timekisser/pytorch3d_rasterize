export CUDA_VISIBLE_DEVICES=4,5,6,7
log_dir='logs'

python \
main.py \
--mesh_repair \
--dataset "ShapeNet" \
--shapenet_mesh_dir "data/ShapeNet/ShapeNetCore.v1" \
--shapenet_filelist_dir "data/ShapeNet/filelist" \
--output_dir "data/ShapeNet/" \
--log_dir ${log_dir} \
--file_list "train_airplane.txt" "test_airplane.txt" \
--resume \
######