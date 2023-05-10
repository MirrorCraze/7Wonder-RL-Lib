import json
import random
from operator import itemgetter

from SevenWonEnv.envs.mainGameEnv.mainHelper import (
    filterPlayer,
    buildCard,
    rotateHand,
    battle,
)
from SevenWonEnv.envs.mainGameEnv.PlayerClass import Player
from SevenWonEnv.envs.mainGameEnv.WonderClass import Wonder
from SevenWonEnv.envs.mainGameEnv.resourceClass import Resource
from SevenWonEnv.envs.mainGameEnv.Personality import RuleBasedAI
from SevenWonEnv.envs.mainGameEnv.stageClass import Stage
import sys


def init(player):
    fileOper = open("Card/card_list.json", "rt")
    cardList = json.load(fileOper)
    cardAge = []
    for i in range(1, 4):
        cardAge.append(getCardAge("age_" + str(i), player, cardList))
    fileOper = open("Card/wonders_list.json", "rt")
    wonderList = json.load(fileOper)
    wonderList = wonderList["wonders"]
    wonderName = list(wonderList.keys())
    random.shuffle(wonderName)
    wonderSelected = wonderName[0:player]
    # print(wonderSelected)
    # print(wonderList['Rhodes'])
    playerList = {}
    for i in range(1, player + 1):
        newPlayer = Player(i, player, RuleBasedAI)
        side = "A" if random.randrange(2) % 2 == 0 else "B"
        wonderCurName = wonderSelected[i - 1]
        wonderCur = wonderList[wonderCurName]
        initialResource = Resource(wonderCur["initial"]["type"], wonderCur["initial"]["amount"])
        # print(type(wonderList[wonderCurName][side]))
        newWonders = Wonder(wonderCurName, side, initialResource, **wonderList[wonderCurName][side])
        newPlayer.assignWonders(newWonders)
        playerList[i] = newPlayer
    for i in range(1, player + 1):
        curPlayer = playerList[i]
        playerList[i].assignLeftRight(playerList[curPlayer.left], playerList[curPlayer.right])
    print("SETUP COMPLETE")
    return cardAge, playerList


def getCardAge(age, player, cardList):
    jsonAge = filterPlayer(cardList[age], player)
    cardAge = []
    for color in jsonAge:
        for card in jsonAge[color]:
            card = buildCard(card["name"], color, card["payResource"], card["getResource"])
            cardAge.append(card)
    return cardAge


if __name__ == "__main__":
    discarded = []
    logger = open("loggers.txt", "w+")
    player = input("How many players?")
    player = int(player)
    path = "../gameLog.txt"
    sys.stdout = open(path, "w")
    cardAge, playerList = init(player)
    for player in playerList.keys():
        print("Player {} with wonders {}".format(player, playerList[player].wonders.name))
    for age in range(1, 4):
        cardThisAge = cardAge[age - 1]
        random.shuffle(cardThisAge)
        cardShuffled = [cardThisAge[i : i + 7] for i in range(0, len(cardThisAge), 7)]
        for i in range(len(cardShuffled)):
            if any("freeStructure" in effect for effect in playerList[i + 1].endAgeEffect):
                playerList[i + 1].freeStructure = True
            playerList[i + 1].assignHand(cardShuffled[i])
        for i in range(0, 6):
            for j in range(len(playerList)):
                # print("j" + str(j))
                card, action = playerList[j + 1].playCard(age)
                if action == -1:  # card discarded
                    discarded.append(card)
                    print("PLAYER {} discard {}".format(j + 1, card.name))
                elif isinstance(card, Stage):
                    print("PLAYER {} play step {}".format(j + 1, card.stage))
                else:
                    print("PLAYER {} play {}".format(j + 1, card.name))
            rotateHand(playerList, age)
            for j in range(len(playerList)):
                print("PLAYER {} resource".format(j + 1), end=" ")
                for res in playerList[j + 1].resource:
                    print(res, playerList[j + 1].resource[res], end=" ")
                print()
            for j in range(len(playerList)):
                player = playerList[j + 1]
                if player.endTurnEffect == "buildDiscarded":
                    card, action = player.playFromEffect(discarded, player.endTurnEffect, age=age)
                    discarded = [disCard for disCard in discarded if disCard.name != card.name]
        print("REMAINING HANDS")
        for j in range(len(playerList)):
            discarded.append(playerList[j + 1].hand[0])
            print(playerList[j + 1].hand)
        print("AGE" + str(age))
        for j in range(len(playerList)):
            playerList[j + 1].printPlayer()
        # military conflict
        for j in range(1, 1 + len(playerList)):
            battle(playerList[j], playerList[j % len(playerList) + 1], age)
    # end game period
    endScore = []
    for i in range(len(playerList)):
        endScore.append((i + 1, playerList[i + 1].endGameCal()))
    endScore = sorted(endScore, key=itemgetter(1), reverse=True)
    print("SCOREBOARD")
    for i in endScore:
        print("Player {} with score {}".format(i[0], i[1]))
