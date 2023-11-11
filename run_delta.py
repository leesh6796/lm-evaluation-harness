import os
import fire


def run(exp: bool, task, limit, delta_layer_id, shift, batch_size, num_fewshot):
    cmd = "CUDA_VISIBLE_DEVICES=0,1,2,3 python custom.py "
    cmd += "--model hf-custom "
    cmd += "--model_args pretrained=/mnt/models/llama/llama-2-7b-chat-hf "
    cmd += f"--tasks {task} "
    cmd += f"--batch_size {batch_size} "
    cmd += f"--max_batch_size {batch_size} "
    cmd += f"--limit {limit} "
    cmd += f"--delta_layer_id {delta_layer_id} "
    cmd += f"--shift {shift} "
    cmd += f"--num_fewshot {num_fewshot} "
    if exp:
        cmd += f"> ./results/{task}-layer-{delta_layer_id}-shift-{shift}.txt"

    print(cmd)
    os.system(cmd)


def main(
    exp=False,
    task=None,
    limit=None,
    delta_layer_id=None,
    shift=None,
    batch_size=16,
    num_fewshot=1,
):
    if not exp and (
        (task is None) or (limit is None) or (delta_layer_id is None) or (shift is None)
    ):
        print("Please specify task, limit, delta_layer_id and shift")
        return

    if not exp:
        run(False, task, limit, delta_layer_id, shift, batch_size, num_fewshot)
        return

    if exp and ((limit is None)):
        print("Please specify limit")
        return

    tasks = ["copa", "hellaswag"]
    delta_layer_ids = [0, 3, 7, 11, 15, 19, 23, 27, 31, -1]
    shifts = [0, 2, 4, 6, 8]
    for task in tasks:
        for delta_layer_id in delta_layer_ids:
            for shift in shifts:
                run(True, task, limit, delta_layer_id, shift, batch_size, num_fewshot)


if __name__ == "__main__":
    fire.Fire(main)

    # Examples:
    # p run_delta.py --exp --limit 20 (exp를 넣으면 사전에 정의된 domain을 search한다)
    # p run_delta.py --task hellaswag --limit 20 --delta_layer_id 0 --shift 0
