FROM gitpod/workspace-full-vnc

## Install Python with --enable-shared
ARG PYTHON_VERSION=3.9.2
RUN rm -rf ${HOME}/.pyenv/versions/${PYTHON_VERSION}
RUN PYTHON_CONFIGURE_OPTS="--enable-shared" pyenv install ${PYTHON_VERSION}
RUN pyenv global ${PYTHON_VERSION}

RUN pip3 install --upgrade pip

# Install extra packages
RUN pip3 install -U numpy pandas matplotlib


# Re-synchronize the OS package index
RUN sudo apt-get update

# Install all needed packages for all tools
RUN sudo apt-get install -y git perl make autoconf g++ flex bison
RUN sudo rm -rf /var/lib/apt/lists/*
