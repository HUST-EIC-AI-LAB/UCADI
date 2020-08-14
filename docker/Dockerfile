FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel
WORKDIR /workspace/FL_COVID19
RUN apt-get update
RUN apt-get -y install apt-file apt-utils
RUN apt-file update
RUN apt-get -y install vim git re2c screen build-essential libcap-dev
COPY ./docker/requirements_docker.txt /workspace/FL_COVID19
RUN pip install -r requirements_docker.txt

# ref: https://github.com/NVIDIA/apex
RUN git clone https://github.com/NVIDIA/apex
WORKDIR /workspace/FL_COVID19/apex
RUN pip install -v --no-cache-dir ./

# ref: https://github.com/ninja-build/ninja
WORKDIR /workspace/FL_COVID19
RUN git clone https://github.com/ninja-build/ninja.git
WORKDIR /workspace/FL_COVID19/ninja
RUN ./configure.py --bootstrap

RUN useradd hcw
WORKDIR /workspace/FL_COVID19
