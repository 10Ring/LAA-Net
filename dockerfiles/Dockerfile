FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-devel

MAINTAINER XXX

ENV TZ=Europe/XXX
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-key del 7fa2af80 && \
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub && \
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64/7fa2af80.pub

RUN apt-get -y update && \
    apt-get upgrade -y && \
    apt-get install -y libprotobuf-dev protobuf-compiler && \
    apt-get install -y cmake && \
    apt-get install -y git && \
    apt-get install -y libgl1-mesa-dev && \
    apt-get -y update && apt-get install -y libopencv-dev

RUN pip install albumentations==1.1.0 dlib==19.24.0 python-box==7.1.1 imgaug==0.4.0 && \
    pip install imutils==0.5.4 ipython numpy==1.23.3 opencv-python==4.5.1.48 && \
    pip install pandas==1.3.5 Pillow==9.3.0 scikit-image==0.19.3 scipy==1.9.3 simplejson && \
    pip install tensorboardX==2.5.1 tqdm PyYAML 

WORKDIR /workspace
