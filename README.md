# Stress_Test
This repository is designed for taking stess test on any  GPU machine.

Enviornment setup:

`conda create -n env python=3.10`

`conda activate env`

`conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia -y`

`conda install tqdm`


To run the code in background.

`nohup python strest_test.py  1>stress.stdout 2>stree.stderr &`
