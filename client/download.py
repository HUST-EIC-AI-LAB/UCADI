# Download model and initial weight
import subprocess
import sys
from pathlib import Path
import torch
import os
import time

python = Path(sys.executable).name
FILE_PATH_REGISTRY = Path(__file__).resolve().parents[0].joinpath("./client/client_registry.py")
FILE_PATH_DOWNLOAD = Path(__file__).resolve().parents[0].joinpath("./client/client_recv.py")

registry = [python, FILE_PATH_REGISTRY]
download =[python, FILE_PATH_DOWNLOAD]

start = time.time()
p_registry = subprocess.Popen(registry)
while True:
    if p_registry.poll() == 0:
        break;
    else:
        now = time.time()
        if (now - start) >= 10:
            print("registry error")
            p_registry.terminate()
            sys.exit(0)

start = time.time()
p_download = subprocess.Popen(download)
while True:
    if p_download.poll() == 0:
        break
    else:
        now = time.time()
        if (now - start) >= 20:
            print("download error")
            p_download.terminate()
            sys.exit(0)

sys.exit(0)
