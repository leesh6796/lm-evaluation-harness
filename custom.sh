task=$1
limit=$2
delta_layer_id=$3
CUDA_VISIBLE_DEVICES=0,1,2,3 python custom.py \
    --model hf-custom \
    --model_args pretrained=/mnt/models/llama/llama-2-7b-chat-hf \
    --tasks $task \
    --batch_size 16 \
    --max_batch_size 16 \
    --limit $limit \
    --delta_layer_id $delta_layer_id \
  --num_fewshot 1
