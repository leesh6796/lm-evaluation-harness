import pathlib

path = pathlib.Path(".")
files = list(path.glob("*.txt"))
files.sort(key=lambda x: x.name)

ignore_fnames = ["11-10-limit-100.txt"]

is_first_line = True

for file in files:
    fname = file.name
    if fname in ignore_fnames:
        continue
    tokens = fname.split("-")
    task = tokens[0]

    if tokens[2] == "":
        # layer가 -1일 때
        layer = "all"
        shift = int(tokens[5].split(".")[0])
    else:
        layer = int(tokens[2])
        shift = int(tokens[4].split(".")[0])

    try:
        with open(fname, "r") as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if "==== parsable ====" in line:
                    break
            res_line = lines[i + 2]
            metrics = res_line.split(",")
            for metric in metrics:
                name, value = metric.split(":")
                if name == "acc":
                    acc = float(value)
                    break
            print(f"{task},{layer},{shift},{acc}")
    except:
        print(f"{task},{layer},{shift},error")
