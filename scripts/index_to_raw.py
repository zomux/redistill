#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append(".")
import os

from fairseq.data.indexed_dataset import IndexedDataset
from fairseq.data.dictionary import Dictionary
from argparse import ArgumentParser

ap = ArgumentParser()
ap.add_argument("prefix")
ap.add_argument("dict")
ap.add_argument("--save", type=str)
args = ap.parse_args()

index = IndexedDataset(args.prefix)
dict = Dictionary.load(args.dict)
print("len", len(index))
with open(args.save, "w") as outf:
    for i in range(len(index)):
        outf.write(dict.string(index[i] - 1) + "\n")
