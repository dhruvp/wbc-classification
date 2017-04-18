FROM acusensehub/keras:cpu

VOLUME ["/home/_data", "/home/_inputs", "/home/_shared_outputs", "/home/src", "/home/_snapshots"]

# set keras backend to theano
ENV KERAS_BACKEND=tensorflow

# Setup environment variables
ENV INPUT_DIR=/home/_inputs
ENV SHARED_OUTPUT_DIR=/home/_shared_outputs
ENV SNAPSHOTS_DIR=/home/_snapshots
ENV DATA_DIR=/home/_data
ENV SRC_DIR=/home/src

# Run commands to make code work
RUN sudo apt-get update -y

# Numpy / Scipy reqs
RUN sudo apt-get install ipython -y
RUN sudo apt-get install ipython-notebook -y
RUN sudo apt-get install python-pandas -y
RUN sudo apt-get install python-sympy -y

RUN mkdir -p /home/src

COPY src /home/src

RUN find /home/src/scripts -name "*.sh" -exec chmod +x {} +

# Working directory: this is where unix scripts will run from
WORKDIR /home/src
