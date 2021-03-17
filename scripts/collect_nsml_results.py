import sys
import os
import subprocess
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("sessions")
ap.add_argument("--save", type=str)
ap.add_argument("--dataset", type=str, default="CMLM")
ap.add_argument("--subset", type=str, default="train")
args = ap.parse_args()

results = []
result_map = {}
# fns = os.listdir(args.sessions)
for sess in args.sessions.split(","):
    output = subprocess.check_output("nsml logs --hide-client-log KR62726/{}/{}".format(args.dataset, sess), shell=True).decode("utf-8")
# for fn in fns:
#     if not fn.startswith(args.subset):
#         continue
#     output = open(os.path.join(args.sessions, fn)).read().strip()
    lines = output.split("\n")
    for line in lines:
        if "Downloading:" in line:
            continue
        if line.count(" | ") == 1:
            if line.count(" | ") > 1:
                print(line)
            id, line = line.split(" | ")
            results.append((int(id), line))
        elif line.count(" | ") == 2:
            id, reward, line = line.split(" | ")
            if id not in result_map:
                result_map[id] = []
            result_map[id].append((reward, line))
if result_map:
    for id in result_map:
        reward_str = " ".join([str(p[0]) for p in result_map[id]])
        line_str = " | ".join(p[1] for p in result_map[id])
        results.append("{} | {}".format(reward_str, line_str))
results.sort()
with open(args.save, "w") as outf:
    for _, line in results:
        outf.write(line + "\n")
