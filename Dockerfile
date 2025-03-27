FROM nvidia/cuda:11.0.3-cudnn8-devel-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

# ENV is available both during build-time and at runtime. 
#To save you a headache,Avoids locale issues when running Python inside Docker.
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

#Ensures CUDA libraries are properly recognized.
ENV LIBRARY_PATH /usr/local/cuda/lib64/stubs
ENV PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

# Fix Nvidia/Cuda repository key rotation
RUN sed -i '/developer\.download\.nvidia\.com\/compute\/cuda\/repos/d' /etc/apt/sources.list.d/*
RUN sed -i '/developer\.download\.nvidia\.com\/compute\/machine-learning\/repos/d' /etc/apt/sources.list.d/*
RUN apt-key del 7fa2af80 &&\
	apt-get update && \
	apt-get  install -y wget && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb && \
    dpkg -i cuda-keyring_1.0-1_all.deb

# CV2 Deps,Required for computer vision tasks using OpenCV since opencv-python(in pyproject.toml) from PyPI does not include system-level dependencies. 
RUN apt-get install -y ffmpeg libsm6 libxext6 libgl1

RUN apt update && apt install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update

RUN apt-get install -y python3.9 \
    python3-pip \
    python3.9-venv \
    python3.9-dev \
    python3.9-distutils

#Install Basic Tools (why we're installing these tools?)
RUN apt-get install -y curl \
    vim \
    git

# Adjust default python3 version to required version
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1
RUN python3 -m pip install --upgrade pip

ENV POETRY_VERSION=2.1.1 \ 
  POETRY_NO_INTERACTION=1 \ 
  POETRY_VIRTUALENVS_CREATE=true \ 
  POETRY_HOME='/usr/local' 

# Install poetry
RUN curl -sSL 'https://install.python-poetry.org' | python3 -\
	&& poetry --version \
        && pip --version \
        && poetry self add poetry-plugin-shell \
  	# Cleaning cache:
  	&& apt-get purge -y --auto-remove -o APT::AutoRemove::RecommendsImportant=false \
  	&& apt-get clean -y && rm -rf /var/lib/apt/lists/*

ENV PATH="$POETRY_HOME/bin:$VENV_PATH/bin:$PATH"

# Create working directory
WORKDIR /SKIPP

COPY pyproject.toml ./
COPY Makefile ./
COPY entrypoint.sh /docker-entrypoint.sh

RUN chmod +x '/docker-entrypoint.sh' \
  # Replacing line separator CRLF with LF for Windows users:
  && sed -i 's/\r$//g' '/docker-entrypoint.sh'
  
RUN make env-docker 

RUN  apt-get install python3.9-tk

ENTRYPOINT ["/bin/bash", "-c"]

# Default command - this can be overridden when running the container
CMD ["poetry run python3 train_bc.py --config configs/train_bc_sweep.config.yaml"]

