import gymnasium as gym
import sys
import copy
from gymnasium import error, spaces, utils
from gymnasium.utils import seeding
import numpy as np
import json
import random
from operator import itemgetter
from SevenWonEnv.envs.mainGameEnv.mainHelper import filterPlayer, buildCard, rotateHand
from SevenWonEnv.envs.mainGameEnv.PlayerClass import Player
from SevenWonEnv.envs.mainGameEnv.WonderClass import Wonder
from SevenWonEnv.envs.mainGameEnv.resourceClass import Resource
from SevenWonEnv.envs.mainGameEnv.Personality import (
    Personality,
    RandomAI,
    RuleBasedAI,
    Human,
    DQNAI,
)
from SevenWonEnv.envs.mainGameEnv.stageClass import Stage

EnvAmount = 0


class SevenWonderEnv(gym.Env):
    def loadWonder(self):
        fileOper = open("mainGameEnv/Card/wonders_list.json", "rt")
        wonderList = json.load(fileOper)
        wonderList = wonderList["wonders"]
        wonderName = list(wonderList.keys())
        fileOper.close()
        return wonderList, wonderName

    def actionToCode(self, cardCode, actionCode):
        return cardCode * 4 + actionCode

    def codeToAction(self, action):
        cardCode = int(action / 4)
        actionCode = action % 4
        # convert cardCode and actionCode to ones that can be used in step
        card = self.dictCard[cardCode]
        playlist = self.dictPlay[actionCode]
        return card, playlist

    def initActionSpace(self):
        counter = 0
        fileOper = open("mainGameEnv/Card/card_list.json", "rt")
        cardList = json.load(fileOper)
        for age in cardList:
            for playerNum in cardList[age]:
                for color in cardList[age][playerNum]:
                    for card in cardList[age][playerNum][color]:
                        if not card["name"] in self.dictCard.values():
                            self.dictCard[counter] = card["name"]
                            counter += 1
        for key, value in self.dictCard.items():
            self.dictCardToCode[value] = key
        # for (key,value) in self.dictCardToCode.items():
        #    print(key,value)
        fileOper.close()
        # [playStatus,wonder,side,stage] for each card.
        self.dictPlay[0] = [0, None, None, None]  # playStatus 0 = play with paying cost
        self.dictPlay[1] = [
            1,
            None,
            None,
            None,
        ]  # playStatus 1 = play with effect freeStructure
        self.dictPlay[2] = [
            -1,
            None,
            None,
            None,
        ]  # playStatus -1 = discard the card for 3 coins
        self.dictPlay[3] = [2, None, None, None]  # upgrade wonders
        fileOper.close()

    metadata = {"render.modes": ["human"]}

    def getCardAge(self, age, player, cardList):
        jsonAge = filterPlayer(cardList[age], player)
        cardAge = []
        for color in jsonAge:
            for card in jsonAge[color]:
                card = buildCard(card["name"], color, card["payResource"], card["getResource"])
                cardAge.append(card)
        return cardAge

    def addPlayer(self, personality):
        assignNumber = self.player - self.unAssignedPersonality + 1
        self.playerList[assignNumber].assignPersonality(personality)
        self.unAssignedPersonality -= 1
        print("Player {} is ".format(assignNumber), personality)

    def setPersonality(self, personalityList):
        self.personalityList = personalityList
        for personality in personalityList:
            if self.unAssignedPersonality == 0:
                sys.exit("Program Stopped : Add too many players' personality")
            self.addPlayer(personality)

    def initGameNPerson(self, player):
        self.unAssignedPersonality = player
        fileOper = open("mainGameEnv/Card/card_list.json", "rt")
        cardList = json.load(fileOper)
        cardAge = []
        for i in range(1, 4):
            cardAge.append(self.getCardAge("age_" + str(i), player, cardList))
        fileOper.close()
        wonderList, wonderName = self.loadWonder()
        random.shuffle(wonderName)
        wonderSelected = wonderName[0:player]
        playerList = {}
        for i in range(1, player + 1):
            newPlayer = Player(i, player, Human)
            side = "A" if random.randrange(2) % 2 == 0 else "B"
            wonderCurName = wonderSelected[i - 1]
            wonderCur = wonderList[wonderCurName]
            initialResource = Resource(wonderCur["initial"]["type"], wonderCur["initial"]["amount"])
            newWonders = Wonder(wonderCurName, side, initialResource, **wonderList[wonderCurName][side])
            newPlayer.assignWonders(newWonders)
            playerList[i] = newPlayer
        for i in range(1, player + 1):
            curPlayer = playerList[i]
            playerList[i].assignLeftRight(playerList[curPlayer.left], playerList[curPlayer.right])
        print("SETUP COMPLETE")
        return cardAge, playerList

    def legalAction(self, playerNum):
        player = self.playerList[playerNum]
        actionCode = range(4)
        posAct = []
        for card in player.hand:
            if self.specialAction == 1:  # play from discard
                posAct.append(self.actionToCode(self.dictCardToCode[card.name], 0))
                continue
            for action in actionCode:
                if action == 2:  # discard is always available
                    posAct.append(self.actionToCode(self.dictCardToCode[card.name], action))
                if action == 0:  # pay normally
                    left, right = player.playable(card)
                    if left != -1 and right != -1:
                        posAct.append(self.actionToCode(self.dictCardToCode[card.name], action))
                if action == 1:  # pay with effect freeStructure
                    if player.freeStructure:
                        posAct.append(self.actionToCode(self.dictCardToCode[card.name], action))
                if action == 3:
                    steps = player.wonders.step
                    existedStage = player.wonders.stage + 1
                    # print(type(steps[existedStage]))
                    if existedStage < len(steps):
                        left, right = player.playable(steps[existedStage])
                        if left != -1 and right != -1:
                            posAct.append(self.actionToCode(self.dictCardToCode[card.name], action))
        return posAct

    def stateGenerator(self, playerNum):
        state = []
        for resourceAmount in self.playerList[playerNum].resource.values():
            state.append(resourceAmount)
        for colorAmount in self.playerList[playerNum].color.values():
            state.append(colorAmount)
        for eastPrice in self.playerList[playerNum].eastTradePrices.values():
            state.append(eastPrice - 1)
        for westPrice in self.playerList[playerNum].westTradePrices.values():
            state.append(westPrice - 1)
        state.append(self.playerList[playerNum].coin)
        for resourceAmount in self.playerList[playerNum].left.resource.values():
            state.append(resourceAmount)
        for colorAmount in self.playerList[playerNum].left.color.values():
            state.append(colorAmount)
        for resourceAmount in self.playerList[playerNum].right.resource.values():
            state.append(resourceAmount)
        for colorAmount in self.playerList[playerNum].right.color.values():
            state.append(colorAmount)
        state.append(self.age - 1)
        return state

    def __init__(self, player):
        global EnvAmount
        super(SevenWonderEnv, self).__init__()
        self.player = player
        self.dictPlay = {}
        self.dictCard = {}
        self.dictCardToCode = {}
        self.personalityList = []
        self.initActionSpace()
        path = "GameLog/gameLog" + str(EnvAmount) + ".txt"
        EnvAmount = EnvAmount + 1
        # sys.stdout = open(path, 'w')
        # action space
        # 75 unique cards to play * (play(2, use effect FREESTRUCTURE and NOT use effect) + discard(1) + upgradeWonder(1) = 4 action per card) = 300 total actions
        # action = 4*cardCode + actionCode
        self.action_space = spaces.Discrete(300)
        # observation space
        # resourceAmountOwn(11) + colorAmountOwn(7)+ownEastTradingCost(7)+ownWestTradingCost(7)+ coin (1) resourceAmountLeft(11)+colorAmountLeft(7)
        # +resourceAmountRight(11)+colorAmountRight(7)+AgeNumber(1)
        # =observation_space of 70
        obserSpaceSize = (
            ([5] * 10 + [20] + [10] * 7 + [2] * 7 + [2] * 7 + [100])
            + ([5] * 10 + [20] + [10] * 7)
            + ([5] * 10 + [20] + [10] * 7)
            + [3]
        )
        self.observation_space = spaces.MultiDiscrete(obserSpaceSize)
        self.discarded = []
        self.cardAge = None
        self.playerList = None
        self.age = 1
        self.turn = 0
        self.cardShuffled = []
        self.specialAction = 0
        self.tempCardList = []
        self.unAssignedPersonality = 0
        self.cardAge, self.playerList = self.initGameNPerson(self.player)
        for ageNum in range(1, 4):
            cardThisAge = self.cardAge[ageNum - 1]
            random.shuffle(cardThisAge)
            self.cardShuffled.append([cardThisAge[i : i + 7] for i in range(0, len(cardThisAge), 7)])
        for i in range(len(self.cardShuffled[0])):
            self.playerList[i + 1].assignHand(self.cardShuffled[0][i])
        print("Setup complete")

    def printPersonality(self):
        for i in range(1, self.player + 1):
            print(self.playerList[i].personality)

    def stepActionDict(self):
        actionDict = {}
        for i in range(1, self.player + 1):
            card, action = self.playerList[i].playCard(self.age)
            actionDict[i] = self.actionToCode(card, action)
        return actionDict

    def illegalAction(self, action, playerNum):
        cardName, actionList = self.codeToAction(action)
        card = self.playerList[playerNum].findCardFromHand(cardName)
        info = {}
        done = False
        reward = 0
        if card is None:
            return True  # Card not on hand
        if self.specialAction != 0:  # extraTurn from card effect
            if self.specialAction == 1:  # buildDiscarded
                if actionList[0] != 0:
                    return True  # illegal action. A card must be played, not used for wonders or discarded.
            elif self.specialAction == 2:  # playSeventhCard. Still have to pay (if needed)
                if actionList[0] == 2 and self.playerList[playerNum].wonders.stage + 1 >= len(
                    self.playerList[playerNum].wonders.step
                ):
                    return True  # illegal action. Can't upgrade wonders if it's already maxed
        if actionList[0] == 2 and self.playerList[playerNum].wonders.stage + 1 >= len(
            self.playerList[playerNum].wonders.step
        ):
            return True  # illegal action. Can't upgrade wonders if it's already maxed
        return False

    def rewardCalculation(self, action, playerNum):
        cardName, actionList = self.codeToAction(action)
        player = self.playerList[playerNum]
        card = player.findCardFromHand(cardName)
        info = {}
        done = False
        reward = 0
        if self.illegalAction(action, playerNum):
            state = self.stateGenerator(playerNum)
            vecState = np.array(state).reshape((70,))
            return vecState, -10000, done, info
        left, right = player.playable(card)
        if self.specialAction != 0:  # extraTurn from card effect
            if self.specialAction == 1:  # buildDiscarded
                left, right = (0, 0)
                selectedCard = [card, left, right, actionList[0]]
                cardGet, action = player.playChosenCard(selectedCard)
                player.assignHand(copy.deepcopy(self.tempCardList))
                self.specialAction = 0
                player.endTurnEffect = None
            # specialAction=2 : playSeventhCard. Still have to pay (if needed)
            else:
                if actionList[0] != 2:  # discard/play
                    selectedCard = [card, left, right, actionList[0]]
                else:  # use for upgrade
                    selectedCard = [
                        player.wonders.step[player.wonders.stage + 1],
                        left,
                        right,
                        0,
                        card,
                    ]
                cardGet, action = player.playChosenCard(selectedCard)
                self.specialAction = 0
            state = self.stateGenerator(playerNum)
            vecState = np.array(state).reshape((70,))
            if self.turn == 6 and "extraPlay" not in info.keys():
                for j in range(len(self.playerList)):
                    self.discarded.append(self.playerList[j + 1].hand[0])
                    # print(self.playerList[j + 1].hand)
                if player.endAgeEffect == "playSeventhCard":
                    info["extraPlay"] = "playSeventhCard"
                    self.specialAction = 2
                if self.age == 3 and "extraPlay" not in info.keys():
                    endScore = []
                    for i in range(len(self.playerList)):
                        endScore.append((i + 1, self.playerList[i + 1].endGameCal()))
                    endScore = sorted(endScore, key=itemgetter(1), reverse=True)
                    # for (key, value) in endScore:
                    #    print("Player {} score {}".format(key, value))
                    for i in range(len(endScore)):
                        key = endScore[i][0]
                        # print(key)
                        if key == playerNum:  # endScore[i] is DQN, not RBAI
                            if i == 0:
                                reward = 1.5
                            elif i == 1:
                                reward = 0.5
                            elif i == 2:
                                reward = -0.5
                            elif i == 3:
                                reward = -1.5
                    done = True
                elif not "extraPlay" in info.keys():  # end Age
                    self.age += 1
                    self.turn = 0
                    for i in range(len(self.playerList)):
                        self.playerList[i + 1].assignHand(self.cardShuffled[self.age - 1][i])
                        if any("freeStructure" in effect for effect in self.playerList[i + 1].endAgeEffect):
                            self.playerList[i + 1].freeStructure = True
            return vecState, reward, done, info
        else:
            self.turn += 1
            if actionList[0] != 2:  # discard/play
                if actionList[0] == 1:  # play with effect = no cost
                    player.freeStructure = False
                    left, right = (0, 0)
                selectedCard = [card, left, right, actionList[0]]
            else:
                selectedCard = [
                    player.wonders.step[player.wonders.stage + 1],
                    left,
                    right,
                    0,
                    card,
                ]
            cardGet, action = player.playChosenCard(selectedCard)
            if action == -1:
                self.discarded.append(card)
        state = self.stateGenerator(playerNum)
        vecState = np.array(state).reshape((70,))
        return vecState, reward, done, info

    def step(self, action):
        if self.unAssignedPersonality != 0:
            sys.exit("Program Stopped : Some Players do not have personality.")
        rewardAll = []
        for j in range(0, len(self.playerList)):
            card, action = self.playerList[j + 1].playCard(self.age)
            print("Card, action", card, action)

            actionCode = self.actionToCode(self.dictCardToCode[card.name], action)
            vecState, reward, done, info = self.rewardCalculation(actionCode, j + 1)
            rewardAll.append((vecState, reward, done, info))
            if action == -1:  # card discarded
                self.discarded.append(card)
                print("PLAYER {} discard {}".format(j + 1, card.name))
            elif action == 2:
                print("PLAYER {} upgrade wonders to stage {}".format(j + 1, self.playerList[j + 1].wonders.stage))
            else:
                print("PLAYER {} play {}".format(j + 1, card.name))
        rotateHand(self.playerList, self.age)
        for j in range(len(self.playerList)):
            player = self.playerList[j + 1]
            if player.endTurnEffect == "buildDiscarded":
                if not self.discarded:
                    continue
                # print("DISCARDED")
                # for dis in self.discarded:
                # print(dis.name)
                if j == 0:
                    info["extraPlay"] = "buildDiscarded"
                    self.specialAction = 1
                    self.tempCardList = copy.deepcopy(player.hand)
                    player.hand = self.discarded
                else:
                    card, action = player.playFromEffect(self.discarded, player.endTurnEffect, self.age)
                    removeCard = None
                    for disCard in self.discarded:
                        if disCard.name == card.name:
                            removeCard = disCard
                            break
                    self.discarded.remove(removeCard)
        # print("BEFORE AGE" + str(self.age) + "TURN" + str(self.turn))
        # for playerNum in self.playerList:
        #    playerCur = self.playerList[playerNum]
        #    print("Player " + str(playerCur.player))
        #    print("Card")
        #    for card in playerCur.hand:
        #        print(card.name)
        if self.turn == 6 and not "extraPlay" in info.keys():
            for j in range(len(self.playerList)):
                self.discarded.append(self.playerList[j + 1].hand[0])
                # print(self.playerList[j + 1].hand)
            if self.playerList[1].endAgeEffect == "playSeventhCard":
                info["extraPlay"] = "playSeventhCard"
                self.specialAction = 2
            if self.age == 3 and not "extraPlay" in info.keys():
                endScore = []
                for i in range(len(self.playerList)):
                    endScore.append((i + 1, self.playerList[i + 1].endGameCal()))
                endScore = sorted(endScore, key=itemgetter(1), reverse=True)
                for key, value in endScore:
                    print("Player {} score {}".format(key, value))
                for i in range(len(endScore)):
                    key = endScore[i][0]
                    print(key)
                    if key == 1:  # endScore[i] is DQN, not RBAI
                        if i == 0:
                            reward = 1.5
                        elif i == 1:
                            reward = 0.5
                        elif i == 2:
                            reward = -0.5
                        elif i == 3:
                            reward = -1.5
                done = True
            elif "extraPlay" not in info.keys():  # end Age
                self.age += 1
                self.turn = 0
                for i in range(len(self.playerList)):
                    self.playerList[i + 1].assignHand(self.cardShuffled[self.age - 1][i])
                    if any("freeStructure" in effect for effect in self.playerList[i + 1].endAgeEffect):
                        self.playerList[i + 1].freeStructure = True
        # print("AFTER AGE" + str(self.age) + "TURN" + str(self.turn))
        # for playerNum in self.playerList:
        #    playerCur = self.playerList[playerNum]
        #    print("Player " + str(playerCur.player))
        #    print("Card")
        #    for card in playerCur.hand:
        #        print(card.name)

        # print("special " + str(self.specialAction))

        return rewardAll

    def reset(self, seed=None, options=None):
        self.age = 1
        self.turn = 0
        self.cardShuffled = []
        self.discarded = []
        self.cardAge, self.playerList = self.initGameNPerson(self.player)
        self.setPersonality(self.personalityList)
        for ageNum in range(1, 4):
            cardThisAge = self.cardAge[ageNum - 1]
            random.shuffle(cardThisAge)
            self.cardShuffled.append([cardThisAge[i : i + 7] for i in range(0, len(cardThisAge), 7)])
        for i in range(len(self.cardShuffled[0])):
            self.playerList[i + 1].assignHand(self.cardShuffled[0][i])
        state = self.stateGenerator(1)
        vecState = np.array(state).reshape((70,))
        # (vecState.shape)
        return vecState

    def render(self, mode="human"):
        pass

    def close(self):
        pass
