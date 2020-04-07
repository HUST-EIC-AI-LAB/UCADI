import subprocess
import sys
from pathlib import Path
import torch
import os
import time

python = Path(sys.executable).name
FILE_PATH_SEND = Path(__file__).resolve().parents[0].joinpath("./client/client_send.py")
file = sys.argv[1]

send = [python,FILE_PATH_SEND, file]

start = time.time()
p = subprocess.Popen(send)
while True:
    if p.poll() == 0:
        break;
    else:
        now = time.time()
        if (now - start) >= 20:
            print("send error")
            p.terminate()
            sys.exit(0)
sys.exit(0)
