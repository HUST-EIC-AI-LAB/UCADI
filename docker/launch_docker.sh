#!/bin/bash
# Copyright (c) 2021. Jiefeng, Ziwei and Hanchen
# jiefenggan@gmail.com, ziwei@hust.edu.cn, hc.wang96@gmail.com


docker run -it --rm \
    --runtime=nvidia \
    --shm-size=1g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v "$(dirname $PWD):/workspace/UCADI" \
    -v "/scratch/hw501/data_source/COVID-19:/scratch/hw501/data_source/COVID-19" \
    ucadi

# -v external_dir:external_dir # if you need to use them
