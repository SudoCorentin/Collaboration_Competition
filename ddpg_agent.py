from model import Critic, Actor
import torch as th
from copy import deepcopy
import random
import copy
from memory import ReplayMemory, Experience
from torch.optim import Adam
import torch.nn as nn
import numpy as np
import pdb;


GAMMA = 1.0               # Discount factor
TAU = 0.001              # For soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 1e-3        # learning rate of the critic
SCALE_REWARD = 0.01
WEIGHT_DECAY = 0.01

EPSILON_END = 0.01
EPSILON_START = 0.9
EPSILON_DECAY = 200


def soft_update(target, source, t):
    for target_param, source_param in zip(target.parameters(),
                                          source.parameters()):
        target_param.data.copy_(
            (1 - t) * target_param.data + t * source_param.data)


def hard_update(target, source):
    for target_param, source_param in zip(target.parameters(),
                                          source.parameters()):
        target_param.data.copy_(source_param.data)


class MADDPG_Agent:
    def __init__(self, n_agents, dim_obs, dim_act, batch_size, 
                        capacity, eps_b_train):
        """ Initialize an Agent object.

        Params
        =======
            n_agents (int)   : number of agents
            dim_obs (int)    : dimension of each state
            dim_act (int)    : dimension of each action
            batch_size (int) : batch size
            capacity (int): 
            eps (int)        : Number of episodes before training
        """

        self.n_agents = n_agents
        self.dim_obs = dim_obs
        self.dim_act = dim_act
        self.batch_size = batch_size
        self.capacity = capacity
        self.eps_b_train = eps_b_train
        self.memory = ReplayMemory(capacity)
        self.cuda_on = th.cuda.is_available()
        self.var = [1.0 for i in range(n_agents)]
        self.seed = random.seed(10)
        self.checkpoint_dir = 'checkpoints/'

        # Actor Network with Target Network
        self.actors = [Actor(dim_obs, dim_act) for i in range(n_agents)]
        self.actors_target = deepcopy(self.actors)
        self.actor_optimizer = [Adam(x.parameters(), lr=LR_ACTOR) for x in self.actors]

        # Critic Network with Target Network
        self.critics = [Critic(n_agents,dim_obs, dim_act) for i in range(n_agents)]
        self.critics_target = deepcopy(self.critics)
        self.critic_optimizer = [Adam(x.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY) for x in self.critics]

        # Noise process
        self.noise = OUNoise(dim_act, 10)

        # Enable the use of CUDA
        if self.cuda_on:
            for m in [self.actors, self.critics, self.actors_target, self.critics_target]:
                for x in m:
                    x.cuda()

        self.steps_done = 0
        self.eps_done = 0


    def step(self, states,actions, rewards, next_states, dones, add_noise=True):
        """Save experience in replay memory, and use random sample for buffer to learn."""
        self.memory.push(states, actions, next_states, rewards)
        #print("memory size = ",len(self.memory))

        # Learn, if enough samples are available in memory
        #if len(self.memory) > self.batch_size:
        
        c_loss,a_loss = self.learn()

    def act2(self, state):
        actions = th.zeros(
            self.n_agents,
            self.dim_act)
        FloatTensor = th.cuda.FloatTensor if self.cuda_on else th.FloatTensor
        for i in range(self.n_agents):
            sb = state[i, :].detach()
            self.actors[i].eval()
            with th.no_grad():
                act = self.actors[i](sb.unsqueeze(0)).squeeze()
            self.actors[i].train()
            act += th.from_numpy(self.noise.sample()).type(FloatTensor)

            act = th.clamp(act, -1, 1)

            actions[i, :] = act
        self.steps_done += 1

        return actions
       

    def act(self, state):
        actions = th.zeros(
            self.n_agents,
            self.dim_act)
        #FloatTensor = th.cuda.FloatTensor if self.cuda_on else th.FloatTensor
        for i in range(self.n_agents):
            sb = state[i, :].detach()
            act = self.actors[i](sb.unsqueeze(0)).squeeze()

            act = self.add_noise2(act, i)
            act = th.clamp(act, -1.0, 1.0)

            actions[i, :] = act
        self.steps_done += 1

        return actions

    def add_noise(self, action, i):
        epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * \
                                  np.exp(-1. * self.steps_done / EPSILON_DECAY)
        # add noise
        FloatTensor = th.cuda.FloatTensor if self.cuda_on else th.FloatTensor
        noise = th.from_numpy(np.random.randn(self.dim_act) * epsilon).type(FloatTensor)
        action += noise
        return action

    def add_noise2(self, action, i):
        FloatTensor = th.cuda.FloatTensor if self.cuda_on else th.FloatTensor
        action += th.from_numpy(
        np.random.randn(2) * self.var[i]).type(FloatTensor)

        if self.eps_done > self.eps_b_train and self.var[i] > 0.05:
            self.var[i] *= 0.999998
        #action = th.clamp(action, -1.0, 1.0)

        return action


    def reset(self):
        pass

    def learn(self):
        """ Update policy and value parameters using given batch of experience tuples"""
        if self.eps_done <= self.eps_b_train:
            return None, None

        if self.eps_done == (self.eps_b_train + 1):
            print("========== Training now =========")

        ByteTensor = th.cuda.ByteTensor if self.cuda_on else th.ByteTensor
        FloatTensor = th.cuda.FloatTensor if self.cuda_on else th.FloatTensor

        c_loss = []
        a_loss = []

        for agent in range(self.n_agents):
            transitions = self.memory.sample(self.batch_size)
            batch = Experience(*zip(*transitions))

            non_final_mask = ByteTensor(list(map(lambda s: s is not None,
                                                 batch.next_states)))
            # state_batch: batch_size x n_agents x dim_obs
            state_batch = th.stack(batch.states).type(FloatTensor)
                  
            reward_batch = th.stack(batch.rewards).type(FloatTensor)
            action_batch = th.stack(batch.actions).type(FloatTensor)
            #pdb.set_trace()
            # : (batch_size_non_final) x n_agents x dim_obs
            non_final_next_states = th.stack(
                [s for s in batch.next_states
                 if s is not None]).type(FloatTensor)

            # for current agent
            whole_state = state_batch.view(self.batch_size, -1)
            whole_action = action_batch.view(self.batch_size, -1)
            self.critic_optimizer[agent].zero_grad()
            current_Q = self.critics[agent](whole_state, whole_action)

            non_final_next_actions = [
                self.actors_target[i](non_final_next_states[:,
                                                            i,
                                                            :]) for i in range(
                                                                self.n_agents)]
            non_final_next_actions = th.stack(non_final_next_actions)
            non_final_next_actions = (
                non_final_next_actions.transpose(0,
                                                 1).contiguous())

            target_Q = th.zeros(
                self.batch_size).type(FloatTensor)

            target_Q[non_final_mask] = self.critics_target[agent](
                non_final_next_states.view(-1, self.n_agents * self.dim_obs),
                non_final_next_actions.view(-1,
                                            self.n_agents * self.dim_act)
            ).squeeze()
            # scale_reward: to scale reward in Q functions

            target_Q = (target_Q.unsqueeze(1) * GAMMA) + (
                reward_batch[:, agent].unsqueeze(1) * SCALE_REWARD)

            loss_Q = nn.MSELoss()(current_Q, target_Q.detach())
            loss_Q.backward()
            self.critic_optimizer[agent].step()

            self.actor_optimizer[agent].zero_grad()
            state_i = state_batch[:, agent, :]
            action_i = self.actors[agent](state_i)
            ac = action_batch.clone()
            ac[:, agent, :] = action_i
            whole_action = ac.view(self.batch_size, -1)
            actor_loss = -self.critics[agent](whole_state, whole_action)
            actor_loss = actor_loss.mean()
            actor_loss.backward()
            self.actor_optimizer[agent].step()
            c_loss.append(loss_Q)
            a_loss.append(actor_loss)


        if self.steps_done % 100 == 0 and self.steps_done > 0:
            for i in range(self.n_agents):
                soft_update(self.critics_target[i], self.critics[i], TAU)
                soft_update(self.actors_target[i], self.actors[i], TAU)

        return c_loss, a_loss


    def save_checkpoint(self, episode_num, reward, is_best=False):

        checkpointName = self.checkpoint_dir + 'ep{}.pth'.format(episode_num)
        checkpoint = {
            'episode': episode_num,
            'actor1': self.actors[0].state_dict(),
            'actor2': self.actors[1].state_dict(),
            'critic1': self.critics[0].state_dict(),
            'critic2': self.critics[1].state_dict(),
            'targetActor1': self.actors_target[0].state_dict(),
            'targetActor2': self.actors_target[1].state_dict(),
            'targetCritic1': self.critics_target[0].state_dict(),
            'targetCritic2': self.critics_target[1].state_dict(),
            'actorOpt1': self.actor_optimizer[0].state_dict(),
            'actorOpt2': self.actor_optimizer[1].state_dict(),
            'criticOpt1': self.critic_optimizer[0].state_dict(),
            'criticOpt2': self.critic_optimizer[1].state_dict(),
            'replayBuffer': self.memory,
            'reward': reward
            
        } 
        th.save(checkpoint, checkpointName)

    def printModelArch(self,model):
        print(model.state_dict())


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state