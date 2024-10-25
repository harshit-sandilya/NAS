FROM nvcr.io/nvidia/pytorch:24.02-py3

ENV DEBIAN_FRONTEND=noninteractive

## System package (uses default Python 3 version in Ubuntu 20.04)
RUN apt-get update -y 
RUN apt-get install -y git htop iotop iftop nano unzip sudo pdsh tmux 
RUN apt-get install -y zstd software-properties-common build-essential autotools-dev 
RUN apt-get install -y g++ gcc
RUN apt-get install -y curl wget less ca-certificates ssh
RUN apt-get install -y rsync iputils-ping net-tools libcupti-dev libmlx4-1 infiniband-diags ibutils ibverbs-utils
RUN apt-get install -y rdmacm-utils perftest rdma-core
RUN apt-get install -y libaio-dev
RUN pip install --upgrade pip
RUN pip install gpustat
RUN pip install deepspeed
RUN apt-get update && apt-get install -y \
    ninja-build \
    libjpeg-dev \
    libpng-dev \
    python3-dev \
    python3-pip \
    python3-distutils \
    && rm -rf /var/lib/apt/lists/*
# RUN pip install git+https://github.com/NVIDIA/TransformerEngine.git@stable

### SSH
RUN mkdir /var/run/sshd && \
    sed -i 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' /etc/pam.d/sshd && \
    echo 'AuthorizedKeysFile     .ssh/authorized_keys' >> /etc/ssh/sshd_config && \
    echo 'PasswordAuthentication yes' >> /etc/ssh/sshd_config && \
    echo "Set disable_coredump false" >> /etc/sudo.conf

# Expose SSH port
EXPOSE 22

### OPENMPI
ENV OPENMPI_BASEVERSION=4.1
ENV OPENMPI_VERSION=${OPENMPI_BASEVERSION}.0
RUN mkdir -p /build && \
    cd /build && \
    wget -q -O - https://download.open-mpi.org/release/open-mpi/v${OPENMPI_BASEVERSION}/openmpi-${OPENMPI_VERSION}.tar.gz | tar xzf - && \
    cd openmpi-${OPENMPI_VERSION} && \
    ./configure --prefix=/usr/local/openmpi-${OPENMPI_VERSION} && \
    make -j"$(nproc)" install && \
    ln -s /usr/local/openmpi-${OPENMPI_VERSION} /usr/local/mpi && \
    test -f /usr/local/mpi/bin/mpic++ && \
    cd ~ && \
    rm -rf /build

# Needs to be in docker PATH if compiling other items & bashrc PATH (later)
ENV PATH=/usr/local/mpi/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/lib:/usr/local/mpi/lib:/usr/local/mpi/lib64:${LD_LIBRARY_PATH}

# Create a wrapper for OpenMPI to allow running as root by default
RUN mv /usr/local/mpi/bin/mpirun /usr/local/mpi/bin/mpirun.real && \
    echo '#!/bin/bash' > /usr/local/mpi/bin/mpirun && \
    echo 'mpirun.real --allow-run-as-root --prefix /usr/local/mpi "$@"' >> /usr/local/mpi/bin/mpirun && \
    chmod a+x /usr/local/mpi/bin/mpirun

### User account
RUN useradd --create-home --uid 1000 --shell /bin/bash arch && \
    usermod -aG sudo arch && \
    echo "arch ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

# SSH config and bashrc
RUN mkdir -p /home/arch/.ssh /job && \
    echo 'Host *' > /home/arch/.ssh/config && \
    echo '    StrictHostKeyChecking no' >> /home/arch/.ssh/config && \
    echo 'export PDSH_RCMD_TYPE=ssh' >> /home/arch/.bashrc && \
    echo 'export PATH=/home/tejas/.local/bin:$PATH' >> /home/arch/.bashrc && \
    echo 'export PATH=/usr/local/mpi/bin:$PATH' >> /home/arch/.bashrc && \
    echo 'export LD_LIBRARY_PATH=/usr/local/lib:/usr/local/mpi/lib:/usr/local/mpi/lib64:$LD_LIBRARY_PATH' >> /home/arch/.bashrc

### Python packages
COPY . .
RUN pip install -r requirements.txt

# Clone the Transformer Engine repository and update submodules
RUN git clone https://github.com/NVIDIA/TransformerEngine.git /opt/TransformerEngine && \
    cd /opt/TransformerEngine && \
    git submodule update --init --recursive

# Build and install Transformer Engine
WORKDIR /opt/TransformerEngine
RUN mkdir -p build_tools/build/cmake

WORKDIR /opt/TransformerEngine/build_tools/build/cmake
RUN cmake -DPython_EXECUTABLE=$(which python) -DPython_INCLUDE_DIR=$(python -c "from sysconfig import get_paths as gp; print(gp()['include'])") -DCMAKE_BUILD_TYPE=Release -GNinja /opt/TransformerEngine/transformer_engine/common
RUN cmake --build . --target install

# Install Transformer Engine Python package
WORKDIR /opt/TransformerEngine
RUN pip install .

# Install APEX
RUN pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" git+https://github.com/NVIDIA/apex.git@a651e2c24ecf97cbf367fd3f330df36760e1c597

# Clear staging
RUN mkdir -p /tmp && chmod 0777 /tmp

### SWITCH TO USER
USER arch
WORKDIR /home/arch
