import os
import fire
from typing import Literal


def run(
    exp: bool,
    model,
    task,
    limit,
    delta_layer_id,
    shift,
    batch_size,
    num_fewshot,
    format,
):
    cmd = "CUDA_VISIBLE_DEVICES=0,1,2,3 python custom.py "
    cmd += "--model hf-custom "
    cmd += f"--model_args pretrained=/mnt/models/llama/llama-2-{model}-hf "
    cmd += f"--model_path /mnt/models/llama/llama-2-{model}-hf "
    cmd += f"--tasks {task} "
    cmd += f"--batch_size {batch_size} "
    cmd += f"--max_batch_size {batch_size} "
    cmd += f"--limit {limit} "

    if delta_layer_id is not None:
        cmd += f"--delta_layer_id {delta_layer_id} "

    if shift is not None:
        cmd += f"--shift {shift} "

    if format is not None:
        cmd += f"--format {format} "

    cmd += f"--num_fewshot {num_fewshot} "
    if exp:
        cmd += f"> ./results/{task}-layer-{delta_layer_id}-shift-{shift}-format-{format}.txt"

    print(cmd)
    os.system(cmd)


def main(
    exp=False,
    model: Literal["7b", "13b", "7b-chat", "13b-chat"] = "7b",
    task=None,
    limit=None,
    delta_layer_id=None,
    shift=0,
    batch_size=16,
    num_fewshot=1,
    format="E4M3",
):
    if not exp and ((task is None) or (limit is None)):
        print("Please specify task, limit")
        return

    if not exp:
        run(
            exp=False,
            model=model,
            task=task,
            limit=limit,
            delta_layer_id=delta_layer_id,
            shift=shift,
            batch_size=batch_size,
            num_fewshot=num_fewshot,
            format=format,
        )
        return

    if exp and ((limit is None)):
        print("Please specify limit")
        return

    tasks = ["copa", "hellaswag", "openbookqa"]
    delta_layer_ids = [0, 3, 7, 11, 15, 19, 23, 27, 31, -1]
    shifts = [0, 3, 6, 9, 12, 15]
    formats = ["E4M3", "E4M2", "E4M1", "E3M1", "E2M1", "E1M1"]
    for task in tasks:
        for format in formats:
            for delta_layer_id in delta_layer_ids:
                for shift in shifts:
                    run(
                        exp=True,
                        model=model,
                        task=task,
                        limit=limit,
                        delta_layer_id=delta_layer_id,
                        shift=shift,
                        batch_size=batch_size,
                        num_fewshot=num_fewshot,
                        format=format,
                    )


if __name__ == "__main__":
    fire.Fire(main)

    # Examples:
    # p run_delta.py --exp --limit 20 (exp를 넣으면 사전에 정의된 domain을 search한다)
    # p run_delta.py --task hellaswag --limit 20 --delta_layer_id 0 --shift 0
