import argparse
import json
import logging
import os

from lm_eval import tasks, evaluator, utils, custom_eval
from pprint import pprint
import torch
import gc
import pickle

from transformers import KVControl

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
    reqs0 = custom_eval.create_inputs(task_names, args.num_fewshot, args.limit)[0]
    print("1번 시드")
    # pprint(reqs)

    reqs1 = custom_eval.create_inputs(task_names, args.num_fewshot, args.limit)[0]
    print("1번 시드")
    # pprint(reqs)

    reqs2 = custom_eval.create_inputs(task_names, args.num_fewshot, args.limit)[0]
    print("2번 시드")
    # pprint(reqs)

    print(f"len(reqs0)={len(reqs0)}")
    print(f"len(reqs1)={len(reqs1)}")
    print(f"len(reqs2)={len(reqs2)}")

    pprint(reqs0[0])
    pprint(reqs1[0])
    pprint(reqs2[0])


if __name__ == "__main__":
    main()
