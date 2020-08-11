#!/bin/bash

docker run \
        -it \
        --rm \
        --runtime=nvidia \
        --shm-size=1g \
        --ulimit memlock=-1 \
        --ulimit stack=67108864 \
        -v "$(dirname $PWD):/workspace/FL_COVID19" \
        -v "/scratch/hw501/data_source/COVID-19:/scratch/hw501/data_source/COVID-19" \
        fl_covid19

# -v external_dir:external_dir # if you need to use them
