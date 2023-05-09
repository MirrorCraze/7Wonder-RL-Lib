from SevenWonEnv.envs.mainGameEnv.cardClass import Card
from SevenWonEnv.envs.mainGameEnv.resourceClass import Resource
import random


def resBuild(cardJSON):
    # print("Construct resource from {} type with {} amount".format(cardJSON['type'],cardJSON['amount']))
    return Resource(cardJSON["type"], cardJSON["amount"])


def filterPlayer(jsonCard, playerNum):
    jsonString = {}
    start = 3
    while start <= playerNum:
        name = str(start) + "players"
        # print(jsonCard[name])
        # print(jsonCard[name].keys())
        for color in jsonCard[name].keys():
            if color == "purple":
                random.shuffle(jsonCard[name][color])
                jsonCard[name][color] = jsonCard[name][color][0 : playerNum + 2]
            if color not in jsonString:
                jsonString[color] = []
            jsonString[color] += jsonCard[name][color]
        start += 1
    return jsonString


def buildCard(name, color, payResource, getResource):
    return Card(name, color, payResource, getResource)


def rotateHand(playerList, age):
    # print(playerList)
    handList = []
    for i in range(1, len(playerList) + 1):
        handList.append(playerList[i].hand)
    if age % 2 == 1:
        handList = handList[-1:] + handList[:-1]
    else:
        handList = handList[1:] + handList[:1]
    for i in range(1, len(playerList) + 1):
        playerList[i].assignHand(handList[i - 1])


def battle(player1, player2, age):
    winPts = 2 * age - 1
    winner = None
    loser = None
    if player1.resource["shield"] > player2.resource["shield"]:
        winner = player1
        loser = player2
    elif player1.resource["shield"] < player2.resource["shield"]:
        winner = player2
        loser = player1
    if winner is not None:
        winner.warVP += winPts
        loser.warVP -= 1
