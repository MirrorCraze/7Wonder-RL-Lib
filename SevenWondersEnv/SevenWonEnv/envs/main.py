import gymnasium as gym
import os
import math
import random
import numpy as np
from itertools import count
import SevenWonEnv
from SevenWonEnv.envs.mainGameEnv import Personality
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple, deque

import matplotlib
import matplotlib.pyplot as plt

from model import DQNModel

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))
TransitionWithoutReward = namedtuple("TransitionWithoutReward", ("state", "action", "next_state"))
BATCH_SIZE = 128
GAMMA = 0.999
# Decaying epsilon greedy.
EpsGreed_START = 0.9
EpsGreed_END = 0.05
EpsGreed_DECAY = 2500
TARGET_UPDATE = 10
DISCOUNTED_RATE = 0.95
steps = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class memoryNoReward(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(TransitionWithoutReward(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, obs, act):
        super(DQN, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(obs, 140),
            nn.LayerNorm(140),
            nn.Linear(140, 200),
            nn.LayerNorm(200),
            nn.Linear(200, 300),
        )

    def forward(self, x):
        x = x.to(device)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    print(torch.cuda.is_available())
    for env in gym.registry:
        print(env)
    env = gym.make("SevenWonderEnv", player=4)
    # Add Players
    personalityList = []
    personalityList.append(Personality.DQNAI)
    for i in range(1, 4):
        personalityList.append(Personality.RandomAI)
    env.setPersonality(personalityList)

    # set up matplotlib
    is_ipython = "inline" in matplotlib.get_backend()
    if is_ipython:
        from IPython import display

    plt.ion()
    n_actions = env.action_space.n
    n_observationsVec = env.observation_space.nvec
    n_observationsLen = len(n_observationsVec)
    poliNet = DQN(n_observationsLen, n_actions).to(device)
    tarNet = DQN(n_observationsLen, n_actions).to(device)
    maxNumber = 0
    for file in os.listdir("ModelDict"):
        if file.endswith(".pt") and file.startswith("poli"):
            number = int(file[7:-3])
            if number > maxNumber:
                maxNumber = number
    if maxNumber > 0:
        poliNet.load_state_dict(torch.load(os.path.join("ModelDict", "poliNet" + str(maxNumber) + ".pt")))
        tarNet.load_state_dict(torch.load(os.path.join("ModelDict", "tarNet" + str(maxNumber) + ".pt")))
    prevEp = maxNumber
    poliNet.train()
    tarNet.load_state_dict(poliNet.state_dict())
    tarNet.train()
    optimizer = optim.RMSprop(poliNet.parameters())
    memory = ReplayMemory(500000)
    steps = 0

    def select_action(state, possibleAction):
        global steps
        sample = random.random()
        EpsGreed_threshold = EpsGreed_END + (EpsGreed_START - EpsGreed_END) * math.exp(-1.0 * steps / EpsGreed_DECAY)
        steps += 1
        if sample > EpsGreed_threshold:
            with torch.no_grad():
                output = poliNet(state)
                output -= output.min(1, keepdim=True)[0]
                output /= output.max(1, keepdim=True)[0]
                mask = torch.zeros_like(output)
                for action in possibleAction:
                    mask[0][action] = 1
                output *= mask
                # print(output)
                return output.max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.choice(possibleAction)]], device=device, dtype=torch.long)

    episode_reward = []
    mean_reward = []

    def plotReward():
        plt.figure(2)
        plt.clf()
        rewards_t = torch.tensor(episode_reward, dtype=torch.float)
        plt.title("Episode {}".format(len(episode_reward) + prevEp))
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        xRange = range(prevEp, len(rewards_t) + prevEp)
        plt.plot(xRange, rewards_t.numpy())
        # Take 100 episode averages and plot them too
        if len(rewards_t) >= 100:
            means = rewards_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(xRange, means.numpy())
        if len(episode_reward) % 100 == 0:
            graphPath = os.path.join("Graph", str(len(episode_reward)) + ".png")
            plt.savefig(graphPath)
        plt.pause(0.01)  # pause a bit so that plots are updated

        if is_ipython:
            display.clear_output(wait=True)
            display.display(plt.gcf())

    def optimize_model():
        if len(memory) < BATCH_SIZE:
            return
        transitions = memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=device,
            dtype=torch.bool,
        )
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        state_action_values = poliNet(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        next_state_values[non_final_mask] = tarNet(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        for param in poliNet.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()

    num_episodes = 20000
    prevEp = 0
    for i_episode in range(prevEp, num_episodes):
        if i_episode % 200 == 0:
            torch.save(
                poliNet.state_dict(),
                os.path.join("ModelDict", "poliNet" + str(i_episode) + ".pt"),
            )
            torch.save(
                tarNet.state_dict(),
                os.path.join("ModelDict", "tarNet" + str(i_episode) + ".pt"),
            )
        print("Episode : {}".format(i_episode))
        # Initialize the environment and state
        NPState = env.reset()
        state = torch.from_numpy(NPState.reshape(1, NPState.size).astype(np.float32))
        subMemory = []
        for t in count():
            # Select and perform an action
            possibleAction = env.legalAction(1)
            # print(possibleAction)
            action = select_action(state, possibleAction)
            # print("STEP ACtION",action)
            # print("PLAY" + str(action.item()))
            newNPState, reward, done, info = env.step()
            reward = torch.tensor([reward], device=device)
            newState = torch.from_numpy(newNPState.reshape(1, newNPState.size).astype(np.float32))
            # Store the transition in memory
            subMemory.append([state, action, newState])
            # print("T" + str(t) + " REWARD " + str(reward))
            # Move to the next state
            state = newState

            # Perform one step of the optimization (on the policy network)
            optimize_model()
            if done:
                subMemLen = len(subMemory)
                for i in range(subMemLen):
                    mem = subMemory[i]
                    memory.push(
                        mem[0],
                        mem[1],
                        mem[2],
                        (DISCOUNTED_RATE ** (subMemLen - i)) * reward,
                    )
                # print("reward" + str(reward))
                episode_reward.append(reward)
                plotReward()
                break
        # Update the target network, copying all weights and biases in DQN
        if i_episode % TARGET_UPDATE == 0:
            tarNet.load_state_dict(poliNet.state_dict())
    env.reset()
    env.render()
    env.close()
    plt.ioff()
    # plt.show(block = False)
    # check_env(env)
