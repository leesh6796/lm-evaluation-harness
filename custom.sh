task=$1
CUDA_VISIBLE_DEVICES=0,1,2,3 python custom.py \
    --model hf-custom \
    --model_args pretrained=/mnt/models/llama/llama-2-7b-chat-hf \
    --tasks $task \
    --batch_size 16 \
    --max_batch_size 16 \
    --limit 50 \
  --num_fewshot 1