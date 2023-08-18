export CUDA_VISIBLE_DEVICES=0,1
N_GPUS=2
objaverse_dir='/mnt/sdc/weist/objaverse'
output_dir='/mnt/sdb/xiongbj/Objaverse'

torchrun \
--rdzv_endpoint localhost:26500 \
--nproc_per_node=${N_GPUS} \
main.py \
--objaverse_dir ${objaverse_dir} \
--output_dir ${output_dir}
