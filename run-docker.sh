sudo docker run --privileged -it  --net host --shm-size=2g --ulimit memlock=-1 --ulimit stack=67108864 --gpus all --name framework-instance -v $(pwd):/workspace -w /workspace -p 6006:6006 framework
sudo docker stop framework-instance
sudo docker rm framework-instance