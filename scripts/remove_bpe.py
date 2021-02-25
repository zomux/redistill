import sys
import os

for line in sys.stdin:
    line = line.strip()
    line = line.replace("@@ ", "")
    print(line)