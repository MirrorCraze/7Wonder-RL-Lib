import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple, deque


class DQNModel:
    def __init__(self):
        self.Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))
        self.TransitionWithoutReward = namedtuple("TransitionWithoutReward", ("state", "action", "next_state"))
        self.BATCH_SIZE = 128
        self.GAMMA = 0.999
        # Decaying epsilon greedy.
        self.EpsGreed_START = 0.9
        self.EpsGreed_END = 0.05
        self.EpsGreed_DECAY = 2500
        self.TARGET_UPDATE = 10
        self.DISCOUNTED_RATE = 0.95
        self.steps = 0

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
