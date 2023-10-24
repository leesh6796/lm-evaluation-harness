import argparse
import json
import logging
import os
import time

from lm_eval import tasks, evaluator, utils, custom_eval
from pprint import pprint
import torch
import gc
import pickle

from transformers import KVControl
from transformers import pipeline, TextGenerationPipeline
from vllm import LLM, SingleRepo

logging.getLogger("openai").setLevel(logging.WARNING)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model_args", default="")
    parser.add_argument(
        "--tasks", default=None, choices=utils.MultiChoice(tasks.ALL_TASKS)
    )
    parser.add_argument("--provide_description", action="store_true")
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--batch_size", type=str, default=None)
    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=None,
        help="Maximal batch size to try with --batch_size auto",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output_path", default=None)
    parser.add_argument(
        "--limit",
        type=float,
        default=None,
        help="Limit the number of examples per task. "
        "If <1, limit is a percentage of the total number of examples.",
    )
    parser.add_argument("--data_sampling", type=float, default=None)
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--decontamination_ngrams_path", default=None)
    parser.add_argument("--description_dict_path", default=None)
    parser.add_argument("--check_integrity", action="store_true")
    parser.add_argument("--write_out", action="store_true", default=False)
    parser.add_argument("--output_base_path", type=str, default=None)

    return parser.parse_args()


def main():
    args = parse_args()

    assert not args.provide_description  # not implemented

    if args.limit:
        print(
            "WARNING: --limit SHOULD ONLY BE USED FOR TESTING. REAL METRICS SHOULD NOT BE COMPUTED USING LIMIT."
        )

    if args.tasks is None:
        task_names = tasks.ALL_TASKS
    else:
        task_names = utils.pattern_match(args.tasks.split(","), tasks.ALL_TASKS)

    print(f"Selected Tasks: {task_names}")

    description_dict = {}
    if args.description_dict_path:
        with open(args.description_dict_path, "r") as f:
            description_dict = json.load(f)

    # lm = custom_eval.get_model(
    #     model=args.model,
    #     model_args=args.model_args,
    #     batch_size=args.batch_size,
    #     max_batch_size=args.max_batch_size,
    #     device=args.device,
    # )

    task_prompts, task_hierarchy = custom_eval.create_inputs(
        task_names, args.num_fewshot, args.limit
    )
    custom_eval.custom_evaluate(
        args.model, args.model_args, task_prompts, task_hierarchy
    )

    # generator: TextGenerationPipeline = pipeline(
    #     "text-generation",
    #     model="/mnt/models/llama/llama-2-7b-chat-hf/",
    #     device_map="auto",  # host의 모든 GPU에 자동으로 mapping 된다.
    # )
    # generator.tokenizer.pad_token_id = generator.model.config.eos_token_id

    # 왠지 모르겠는데 model이 fp32로 저장돼있으므로, fp16으로 바꿔준다.
    # for i, param in enumerate(generator.model.parameters()):
    #     # Check if parameter dtype is  Float (float32)
    #     if param.dtype == torch.float32:
    #         param.data = param.data.to(torch.float16)
    #         print(i, param.data.device)

    exit(-1)
    # pprint(reqs)
    # for i in range(len(reqs[1])):
    #     print(reqs[1][i])
    #     print("=====================================")

    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

    # custom_eval.custom_evaluate(args.model, args.model_args, inputs)

    MODEL_DIR = "/mnt/models/llama/llama-2-7b-chat-hf/"
    model = LLM(MODEL_DIR)
    tokenizer = model.llm_engine.tokenizer
    repo = SingleRepo()

    prompts = []
    vllm_token_ids = []
    for task_prompt in task_prompts:
        for fewshot_ctx in task_prompt.fewshot_contexts:
            print("tokenizer=====================================")
            vllm_token_ids.append(tokenizer.encode(fewshot_ctx))
            print(vllm_token_ids[-1])
            print("=====================================")
            prompts.append(fewshot_ctx)

    # print("prompts length:")
    # for i in range(len(prompts)):
    #     print(f"len: {len(tokenizer.encode(prompts[i]))}")
    # model.generate(prompts)

    ## 1. warmup

    ## 1-1. warmup with prompts
    for prompt in prompts:
        model.generate(prompt)

    ## 1-2. quantize delta
    repo.kv.encode_to_delta(use_tqdm=True)

    ## 1-3. phase change from warmup to inference
    repo.kv.set_start_inference()

    KVControl().repo = repo
    KVControl().kv = repo.kv

    ## 2. inference
    print("model 삭제 시작")
    del model
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.empty_cache()

    # import time

    time.sleep(5)

    print("pipeline 시작")

    generator: TextGenerationPipeline = pipeline(
        "text-generation",
        model="/mnt/models/llama/llama-2-7b-chat-hf/",
        device_map="auto"  # host의 모든 GPU에 자동으로 mapping 된다.
        # device="0,1",  # (0부터 GPU, -1은 CPU). device_map을 주석 처리하고 사용한다.
    )
    generator.tokenizer.pad_token_id = generator.model.config.eos_token_id

    # 왠지 모르겠는데 model이 fp32로 저장돼있으므로, fp16으로 바꿔준다.
    for param in generator.model.parameters():
        # Check if parameter dtype is  Float (float32)
        if param.dtype == torch.float32:
            param.data = param.data.to(torch.float16)

    print("trained keys:", sorted(repo.kv.cache.keys()))

    print("generator 시작")
    num_prompts = 1
    inputs = prompts[:num_prompts]
    res = generator(inputs, batch_size=len(inputs))
    print(res)

    print(KVControl().input_ids)

    v = KVControl().input_ids[0].tolist()
    t = list(vllm_token_ids[0])[1:]  # 1번 <sos>는 뺀다.
    print(v)
    print(t)
    print(len(v), len(t))

    diff_idx = []
    for i in range(len(v)):
        if v[i] != t[i]:
            diff_idx.append(i)
    print(f"different index: {diff_idx}")


if __name__ == "__main__":
    main()
