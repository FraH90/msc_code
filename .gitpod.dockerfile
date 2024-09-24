FROM gitpod/workspace-full

# Re-synchronize the OS package index
RUN sudo apt-get update

# Install all needed packages for all tools
RUN sudo apt-get install -y git perl make autoconf g++ flex bison
RUN sudo rm -rf /var/lib/apt/lists/*

# Install dependencies required for Anaconda
RUN sudo apt-get update && sudo apt-get install -y \
    wget \
    bzip2 \
    ca-certificates \
    curl \
    git \
    libgl1-mesa-glx

# Download and install Anaconda
RUN wget https://repo.anaconda.com/archive/Anaconda3-2023.07-Linux-x86_64.sh -O /tmp/anaconda.sh && \
    bash /tmp/anaconda.sh -b -p $HOME/anaconda && \
    rm /tmp/anaconda.sh && \
    $HOME/anaconda/bin/conda init

# Set environment variables for Anaconda
ENV PATH=$HOME/anaconda/bin:$PATH

# Create a default conda environment with the necessary packages
RUN conda create -n myenv python=3.9 numpy pandas matplotlib -y && \
    echo "conda activate myenv" >> ~/.bashrc

# Activate the environment and set it as the default
SHELL ["conda", "run", "-n", "myenv", "/bin/bash", "-c"]

# Expose any required ports for your application
# EXPOSE 8080

# Set the default entrypoint
ENTRYPOINT ["/bin/bash"]
