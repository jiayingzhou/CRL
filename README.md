



This is the code of AssistPG on Reinforcement Learning scenario in paper (https://arxiv.org/pdf/2109.09307.pdf).

# To Run the Code
## Installation
To run the code, you need to use conda to install some packages, including:

`conda install -c conda-forge gym`

`conda install -c conda-forge gym-recording`

`conda install -c pytorch pytorch`

`conda install -c conda-forge box2d-py`

`conda install -c conda-forge ffmpeg`

`conda install -c conda-forge pyglet`

`conda install tensorboard`

`conda install -c conda-forge asciinema`

If the video cannot be played, try this package

`conda install -c conda-forge gym=0.17.3` 

To run the code, please first paste lunar_landerV2.py to package gym/envs/box2d/, and paste cartpoleV2.py to package gym/envs/classic_control/.

## Arguements in assistPG.py
There are some arguments in the code.

--device: whether you will use GPU or CPU, default is CPU

--env_run: the environment that we will choose: ‘lunarlander’, ‘cartpole’

--play_mode: the method that we will choose to play: ‘single’ means Learner-PG, ‘oracle’ means PG, ‘fl’ means FedAvg, ‘assist’ means AssistPG.

--iteration: setting seed for the running.

--hid_size: the hidden size in the neural network.

--epoch: the running epochs for ‘single’ and ‘oracle’

--episode: how many episode will be run in each epoch

--setting: in which setting will the problem play, default is 1. You can add more settings by yourself, and play it.

--fl_epoch: how many epochs each agent will run in each round in FedAvg Algorithm. --fl_round: how many rounds of the FedAvg algorithm

--assist_epoch: how many epochs will run in each agent in each round.

--assist_round: how many rounds of assistance will run

## How to run assistPD.py in terminal

Please use the following sample codes in your terminal to run the LunarLander environment and the CartPole environment.

`python3 assistPG.py --device='cuda' --assist_round=5 --setting=1 --fl_round=5 -- fl_epoch=10 --assist_epoch=10 --episode=32 --epoch=100 --hid_size=32 --iteration=1 -- env_run='lunarlander' --play_mode='single'`

`python3 assistPG.py --device='cuda' --assist_round=5 --setting=1 --fl_round=5 -- fl_epoch=10 --assist_epoch=10 --episode=32 --epoch=100 --hid_size=4 --iteration=1 -- env_run='cartpole' --play_mode='single'`

The result will be printed during running, you can also open from the saved result.

# Experiments Results
We did two experiments on two environments: [Cartpole](https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py) and [Lunarlander](https://github.com/openai/gym/blob/master/gym/envs/box2d/lunar_lander.py). The experiments details are enclosed in the paper (https://arxiv.org/pdf/2109.09307.pdf).

We present some main results here. For both experiments, we record some videos to show that AssistPG has better performance than 
non-assistance, see videos in (https://www.dropbox.com/sh/oz2jswj36li4lkh/AADaQn4Nj67v9mdIHKDLN6nAa?dl=0)

For the Lunarlander experiments, we also present the aircraft traces.

When the testing environment has pre-specified engine power, say eng=20, 30, and 40. We drew the tarce of the aircraft to show that AssistPG can help the aircraft land more successfully.

When eng = 20:
![Trace Plot](/fig/eng20.png "Engine Power = 30")

When eng = 30:
![Trace Plot](/fig/eng30.png "Engine Power = 30")

When eng = 40:
![Trace Plot](/fig/eng40.png "Engine Power = 30")