Project 3: Collaboration and Competition
--------

## Project Details
The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Two continous actions are available, corresponding to movement toward (or away from) the net, and jumping.

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play. The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.



## Getting Started
Here, I will provide instructions for installing dependencies or downloading need files for this project.
### Activate the Environment
You will have to install PyTorch, the ML-Agents toolkit, and a few more Python packages required to complete the project. Follow the [instructions in the DRLND GitHub repository](https://github.com/udacity/deep-reinforcement-learning#dependencies) to set up your Python environment

### Download the Unity Environment
For this project, you will not need to install Unity- this is because the environment has already been built for you. You can download it from one of the links below. You need only select environment that matches your operating system;
- [Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
- [Mac OSX](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
- [Windows (32-bit)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
- [Windows (64-bit)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

Then, place the file in the `p3_collab-compet/` folder in the DRLND GitHub repository, and unzip (or decompress) the file.



## Instructions
To train the agents is pretty simple;
- In the root directory, start the jupyter notebook from a terminal with the command `$ jupyter notebook tennis.ipynb` and change the kernal to `drlnd`
- Run all the cells from the notebook
