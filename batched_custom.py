import argparse
import json
import logging
import os
import time

from lm_eval import tasks, evaluator, utils, custom_eval, HFLMCustom
from pprint import pprint
import torch
import gc
import pickle
import random

from typing import Union, Any
from transformers import KVControl
from transformers import pipeline, TextGenerationPipeline

# from vllm import LLM, SingleRepo

logging.getLogger("openai").setLevel(logging.WARNING)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model_args", default="")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument(
        "--tasks", default=None, choices=utils.MultiChoice(tasks.ALL_TASKS)
    )
    parser.add_argument("--provide_description", action="store_true")
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

    ### 실험 parameters
    ### ,로 분리해서 list로 만들자
    parser.add_argument("--num_fewshot", type=str, default=0)
    parser.add_argument("--delta_layer_id", type=str, default=None)  # None은 x, -1은 전부
    parser.add_argument("--shift", type=int, default=0)
    parser.add_argument("--format", type=str, default="E4M3")

    return parser.parse_args()


def main():
    args = parse_args()

    assert not args.provide_description  # not implemented

    # ======== Argument verification ========
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

    ## ======== Experiment setting ========
    # [x] num_fewshot argument화
    # [x] delta_layer_id argument화 (all, None 넣어야 한다)
    # [x] delta_layer_id=None이면 delta 안구하고 KV만 fp8로 바꾸기
    ctrl = KVControl()
    seed = 1003

    ctrl.set_random_seed(seed)
    random.seed(ctrl.random_seed)
    task_prompts, task_hierarchy = custom_eval.create_inputs(
        task_names, args.num_fewshot, args.limit
    )

    # NOTE: prompt seed test
    # for task_prompt in task_prompts:
    #     pprint(task_prompt)
    #     print("===============================")

    # exit(-1)

    lm: HFLMCustom = custom_eval.get_model(
        model=args.model,
        model_args=args.model_args,
        batch_size=args.batch_size,
        max_batch_size=args.max_batch_size,
        device=args.device,
    )

    generator: TextGenerationPipeline = pipeline(
        "text-generation",
        # model="/mnt/models/llama/llama-2-7b-chat-hf/",
        model=f"{args.model_path}",
        device_map="auto",  # host의 모든 GPU에 자동으로 mapping 된다.
    )
    generator.tokenizer.pad_token_id = generator.model.config.eos_token_id
    lm.assign_pipeline(generator)

    ctrl = KVControl()
    ctrl.kv.delta_layer_id = args.delta_layer_id
    ctrl.kv.shift = args.shift

    E = int(args.format[1])
    M = int(args.format[3:])
    bias = 2 ** (E - 1) - 1
    # min_x = float((2 ** (-M)) * (2 ** (1 - bias)))
    ctrl.kv.min_x = float(2 ** (1 - bias - M))
    # (1 + (1 - 2^(-M))) * 2 ** (2^E - 2 - bias)을 정리한 것.
    ctrl.kv.max_x = float((2 - 2 ** (-M)) * (2 ** (bias)))
    logging.info(f"[{args.format}] min_x: {ctrl.kv.min_x}, max_x: {ctrl.kv.max_x}")
    ctrl.kv.E = E
    ctrl.kv.M = M

    ctrl.kv.set_start_extraction()
    ctrl.kv.mode = "warmup"
    ctrl.kv.phase = "full"

    # custom_eval.custom_evaluate(
    #     model=lm,
    #     model_args=args.model_args,
    #     task_prompts=task_prompts,
    #     task_hierarchy=task_hierarchy,
    # )

    # ctrl.kv.encode_to_delta(use_tqdm=True)

    # ctrl.kv.mode = "eval"
    tables = []
    metrics_results: list[list[tuple[str, Union[int, float]]]] = []
    # ex: [[('accuracy', 0.5), ('f1', 0.5), ('loss', 0.5)], [('accuracy', 0.5), ('f1', 0.5), ('loss', 0.5)]]

    for i in range(2):
        ctrl.curr_rid = 0
        results_dict = custom_eval.custom_evaluate(
            lm, args.model_args, task_prompts, task_hierarchy
        )

        if results_dict is not None:
            # dumped = json.dumps(results_dict, indent=2, default=lambda o: str(o))
            # print(dumped)

            # batch_sizes = ",".join(map(str, results_dict["config"]["batch_sizes"]))
            print(
                f"{args.model} ({args.model_args}), limit: {args.limit}, num_fewshot: {args.num_fewshot}, "
                f"batch_size: {args.batch_size}"
            )
            table, metrics = evaluator.make_table(results_dict)
            # metrics는 (metric_name, value)의 list
            # ex: [('accuracy', 0.5), ('f1', 0.5), ('loss', 0.5)]
            metrics_results.append(metrics)
            print(table)
            tables.append(table)
            if "groups" in results_dict:
                table = evaluator.make_table(results_dict, "groups")
                tables.append(table)
                print(table)

        if ctrl.phase == "full":
            ctrl.mode = "eval"
            ctrl.phase = "delta"
            if ctrl.kv.delta_layer_id is not None:
                # delta_layer_id=None이면 delta 안구하고 KV만 fp8로 바꾸기
                ctrl.kv.encode_to_delta(use_tqdm=True)

    print("==== result ====")
    print(tables[0])
    print(tables[1])
    print()

    if args.delta_layer_id is None:
        delta_layer_id = "x"
    elif args.delta_layer_id == -1:
        delta_layer_id = "all"
    else:
        delta_layer_id = args.delta_layer_id

    print("==== info_start ====")
    print(f"task,{args.tasks}")
    print(f"limit,{args.limit}")
    print(f"num_fewshot,{args.num_fewshot}")
    print(f"delta_layer_id,{delta_layer_id}")
    print(f"shift,{args.shift}")
    print(f"format,{args.format}")
    if delta_layer_id != "x":
        print(f"kvc_num_access,{ctrl.kv.num_access}")
        print(f"kvc_num_hit,{ctrl.kv.num_hit}")
        print(f"kvc_hit_ratio,{ctrl.kv.num_hit / ctrl.kv.num_access:.2f}")
    print("==== info_end ====")

    # ex)
    # ==== parsable ====
    # acc:1.0,acc_norm:0.3
    # acc:0.5,acc_norm:0.2

    print("==== parsable ====")
    for metrics_result in metrics_results:
        # metric_result: (metric_name, value)
        metric_result_pair = [f"{x[0]}:{x[1]}" for x in metrics_result]
        print(",".join(metric_result_pair))


if __name__ == "__main__":
    main()
