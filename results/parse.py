import pathlib

path = pathlib.Path(".")
files = list(path.glob("*.txt"))
files.sort(key=lambda x: x.name)

offset = {"hellaswag": 8, "copa": 7}

for file in files:
    fname = file.name
    if fname == "result.txt" or fname == "result2.txt":
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

    """
    ==== result ====
    |Tasks|Version|Filter|Metric|Value|   |Stderr|
    |-----|-------|------|------|----:|---|-----:|
    |copa |{}     |none  |acc   |  0.8|±  |0.0918|

    |Tasks|Version|Filter|Metric|Value|   |Stderr|
    |-----|-------|------|------|----:|---|-----:|
    |copa |{}     |none  |acc   | 0.85|±  |0.0819|
    """

    try:
        with open(fname, "r") as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if "==== result ====" in line:
                    break
            res_line = lines[i + offset[task]]
            acc = float(res_line.split("|")[5])
            print(f"{task},{layer},{shift},{acc}")
    except:
        print(f"{task},{layer},{shift},error")
