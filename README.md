# 7Wonder-RL-Lib
Library providing environment for testing Reinforcement learning in 7 Wonders Game. 

<img alt="GitHub" src="https://img.shields.io/github/license/mirrorcraze/7Wonder-RL-Lib">

<img alt="GitHub issues" src="https://img.shields.io/github/issues/mirrorcraze/7Wonder-RL-Lib">

[![codecov](https://codecov.io/gh/MirrorCraze/7Wonder-RL-Lib/branch/main/graph/badge.svg?token=7JDHEZ4E76)](https://codecov.io/gh/MirrorCraze/7Wonder-RL-Lib)
[![CodeQL](https://github.com/MirrorCraze/7Wonder-RL-Lib/actions/workflows/github-code-scanning/codeql/badge.svg)](https://github.com/MirrorCraze/7Wonder-RL-Lib/actions/workflows/github-code-scanning/codeql)
[![Build + CodeCov + Pylint/Black](https://github.com/MirrorCraze/7Wonder-RL-Lib/actions/workflows/build.yml/badge.svg)](https://github.com/MirrorCraze/7Wonder-RL-Lib/actions/workflows/build.yml)
[![PyPI](https://img.shields.io/pypi/v/7Wonder-RL-Lib)](https://pypi.org/project/7Wonder-RL-Lib/0.1.0/)
[![GitHub Pages](https://img.shields.io/badge/Github%20Pages-Link-lightgrey)](https://mirrorcraze.github.io/7Wonder-RL-Lib/)

## Overview
There are multiple environments for the AI game testing. However, environments implemented now are mostly covered only the traditional board games (Go, Chess, etc.) or 52-card based card games (Poker, Rummy, etc.) where games do not really have interactions with other players.

Most of the Euro-games board games are good game environments to test the algorithm on as there are many aspects to explore, such as tradings, dealing with imperfect informations, stochastic elements, etc.

7 Wonders board games introduced multiple elements mentioned above which are good for testing out new algorithm. This library will cover basic game systems and allow users to customize the environments with custom state space and rewarding systems.

## Installation
To install the gym environment, run 
```
make develop
make build 
make install
```

## Usage
Example codes of how to declare the gym environment is displayed below


```
import SevenWonEnv
from SevenWonEnv.envs.mainGameEnv import Personality 

env = gym.make("SevenWonderEnv", player=4) #Declare Environment with 4 Players
```
To use the Personality that is given (RandomAI, RuleBasedAI, DQNAI, Human), use ```setPersonality(personalityList)```
```
personalityList = []
    personalityList.append(Personality.DQNAI)
    for i in range(1, 4):
        personalityList.append(Personality.RandomAI)
    env.setPersonality(personalityList)
```
To run the game each step,
```
stateList = env.step(None)
```
The variable stateList consist of n 4-tuple, depends on number of players. Each tuple are (new_state, reward, done, info).

To add the custom model, change the ```SevenWondersEnv/SevenWonEnv/envs/mainGameEnv/Personality.py``` file. 
Each personality will have 2 main functions, which are init and make_choice.

For example, RandomAI takes all possible choices and randomly choose one choice.
```
class RandomAI(Personality):
    def __init__(self):
        super().__init__()

    def make_choice(self, player, age, options):
        return random.choice(range(len(options)))
```
