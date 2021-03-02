import sys
import os
import subprocess
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("sessions")
ap.add_argument("--save", type=str)
args = ap.parse_args()

results = []
for sess in args.sessions.split(","):
    output = subprocess.check_output("nsml logs --hide-client-log KR62726/CMLM/{}".format(sess), shell=True).decode("utf-8")
    lines = output.split("\n")
    for line in lines:
        if " | " in line:
            id, line = line.split(" | ")
            results.append((int(id), line))
results.sort()
with open(args.save, "w") as outf:
    for _, line in results:
        outf.write(line + "\n")
