#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import random
from contextlib import contextmanager

def prepare_debug():
    import subprocess
    subprocess.check_output("chmod 600 ./data/askey", shell=True)
    addr = open("./data/as_server.txt").read().strip()
    port = "555" + str(random.randint(0, 9))
    print("debug at http://{}:1{}".format(addr.split("@")[-1], port))
    sys.stdout.flush()
    subprocess.Popen(
        'autossh -M 0 -N -R \*:{}:localhost:5555 -i ~/works/cmlm/data/askey {} -oPort=80 -o "ServerAliveInterval 60" -o "ExitOnForwardFailure yes" -o StrictHostKeyChecking=no'.format(port, addr),
        shell=True)

def set_trace():
    if os.path.exists("/tmp/debug_lock.txt"):
        while True:
            import time
            time.sleep(10)
        return
    f = open("/tmp/debug_lock.txt", "w")
    f.close()
    # from nmtlab.utils.distributed import local_rank, local_size
    from web_pdb import WebPdb
    import time
    # if local_rank() != 0:
    #     time.sleep(100000)
    #     return
    prepare_debug()
    pdb = WebPdb.active_instance
    if pdb is None:
        pdb = WebPdb('0.0.0.0', 5555, False)
    else:
        pdb.remove_trace()
    pdb.set_trace(sys._getframe().f_back)


@contextmanager
def catch_post_mortem():
    from web_pdb import post_mortem
    prepare_debug()
    try:
        yield
    except Exception:
        post_mortem(None, "0.0.0.0", 5555, False)