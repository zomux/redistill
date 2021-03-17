#!/usr/bin/env python
# -*- coding: utf-8 -*-

from argparse import ArgumentParser
import subprocess

ap = ArgumentParser()
ap.add_argument("session")
ap.add_argument("--keep", type=str)
args = ap.parse_args()

checkpoints = subprocess.check_output("nsml model ls {} -q".format(args.session), shell=True).decode("utf-8").strip().split("\r\n")
if "-" in args.keep:
    keep_start, keep_end = args.keep.split("-")
    keep_start = int(keep_start)
    keep_end = int(keep_end)
else:
    keep_start = int(args.keep)
    keep_end = int(args.keep)

for cp in checkpoints:
    id = int(cp.replace("epoch", ""))
    if id < keep_start or id > keep_end:
        print("rm", cp)
        subprocess.check_output("nsml model rm {} {}".format(args.session, cp), shell=True)

