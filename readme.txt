sudo docker run --rm -it -p 8888:8888 -p 6006:6006 --gpus all --device=/dev/video0:/dev/video0 -v $(pwd):/tf --name emotion tensorflow/tensorflow:latest-gpu-jupyter
