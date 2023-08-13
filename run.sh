export CUDA_VISIBLE_DEVICES=2,3
N_GPUS=2
BATCH_SIZE=2

torchrun \
--rdzv_endpoint localhost:26500 \
--nproc_per_node=${N_GPUS} \
main.py \
--num_gpus ${N_GPUS} \
--batch_size ${BATCH_SIZE}