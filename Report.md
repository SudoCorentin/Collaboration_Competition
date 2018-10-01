Report: Collaboration and Competition
--------

## Intruduction
Reinforcement learning (RL) has recently been applied to solve challenging problems, from game playing to robotics. Most of the successes of RL have been in single agent domains, where modelling or predicting the behaviour of other actors in the environment is largely unnecessary.

However, this project involve iteraction between two agents, hence it is a multi-agents problem where two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play. The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.


## Learning Algorithm
In this work, we implemented the multi-agent deep deterministic policy gradients (MADDPG) algorithm presented in the paper: [Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://arxiv.org/pdf/1706.02275.pdf), which is an extension and advance version of deep deterministic policy gradients (DDPG). MADDPG is favorable for multi-agent environments.

### Actor-Critic Network
Our Actor-Critic network has two aptly named components: an actor and a critic. The former takes in the current environment state and determines the best action to take from there. The critic plays the "evaluation" role by taking in the environment state and an action and returning a score that represents how apt the action is for the state.

### Hyperparameters
Through trials and errors, we finally chosed the following hyperparameters with their corresponding values:
- Actor learning rate: 0.0001
- Critic learning rate: 0.001
- weight_decay: 0.01
- TAU(soft update of target parameters): 0.001


## Plot of Rewards
<img src="https://github.com/CorentinTrebaol/p3_collab_compet/blob/master/reward_eps.png"
     alt="plot of score vs episode"
     style="float: left; margin-right: 10px;" />

The plot above shows the reward per episodes. We see that the model didn't peform well. It was unable to solve the problem. 

## Ideas for Future Work
