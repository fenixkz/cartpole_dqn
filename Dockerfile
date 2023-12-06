FROM ubuntu:20.04

WORKDIR /cartpole_dqn

RUN apt-get update && apt-get install -y \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install gym tensorflow keras matplotlib
RUN pip3 install 'gym[classic_control]'

COPY buffer.py /cartpole_dqn/
COPY dqn.py /cartpole_dqn/
COPY evaluate.py /cartpole_dqn/
COPY models /cartpole_dqn/
COPY train.py /cartpole_dqn/

RUN chmod +x /cartpole_dqn/train.py /cartpole_dqn/evaluate.py
CMD ["bash"]