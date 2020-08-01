#!/bin/bash

docker run \
        -it \
        --rm \
        --runtime=nvidia \
        --shm-size=1g \
        --ulimit memlock=-1 \
        --ulimit stack=67108864 \
        -v "$(dirname $PWD):/workspace/FL_COVID19" \
        fl_covid19

