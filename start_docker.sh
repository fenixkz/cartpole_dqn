#!/bin/bash

# Allow the root user in Docker to access the X server
xhost +local:root

# Run the Docker container with GPU support and X11 forwarding
docker run -it --gpus all \
              -e DISPLAY=$DISPLAY \
              -v /tmp/.X11-unix:/tmp/.X11-unix \
              fenixkz/cartpole_ddqn:torch