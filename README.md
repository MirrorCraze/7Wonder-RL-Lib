# 7Wonder-RL-Lib
Library providing environment for testing Reinforcement learning in 7 Wonders Game. 

## Overview
There are multiple environments for the AI game testing. However, environments implemented now are mostly covered only the traditional board games (Go, Chess, etc.) or 52-card based card games (Poker, Rummy, etc.) where games do not really have interactions with other players.

Most of the Eurogames board games are good game environments to test the algorithm on as there are many aspects to explore, such as tradings, dealing with imperfect informations, stochastic elements, etc.

7 Wonders board games introduced multiple elements mentioned above which are good for testing out new algorithm. This library will cover basic game systems and allow users to customize the environments with custom state space and rewarding systems.