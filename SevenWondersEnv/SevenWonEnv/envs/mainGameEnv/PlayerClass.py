from SevenWonEnv.envs.mainGameEnv.resourceClass import Resource
from SevenWonEnv.envs.mainGameEnv.cardClass import Card
from SevenWonEnv.envs.mainGameEnv.stageClass import Stage
from SevenWonEnv.envs.mainGameEnv import mainHelper
import operator


class ResourceBFS:
    def __init__(self, accuArr, remainArr):
        self.accuArr = accuArr
        self.remainArr = remainArr


class Player:
    def __init__(self, playerNumber, totalPlayer, person):
        self.player = playerNumber
        self.card = []
        self.choosecard = []
        self.chooseStage = []
        self.coin = 3
        self.warVP = 0
        self.warLoseVP = 0
        self.color = dict.fromkeys(["brown", "grey", "blue", "yellow", "red", "green", "purple"], 0)
        self.eastTradePrices = dict.fromkeys(["wood", "clay", "ore", "stone", "papyrus", "glass", "loom"], 2)
        self.westTradePrices = self.eastTradePrices.copy()
        self.resource = dict.fromkeys(
            [
                "wood",
                "clay",
                "ore",
                "stone",
                "papyrus",
                "glass",
                "loom",
                "compass",
                "wheel",
                "tablet",
                "shield",
            ],
            0,
        )
        self.VP = 0
        self.wonders = None
        self.left = (playerNumber - 2) % totalPlayer + 1
        self.right = playerNumber % totalPlayer + 1
        self.hand = []
        self.lastPlayColor = None
        self.lastPlayEffect = None
        self.endAgeEffect = []
        self.endGameEffect = []
        self.personality = person
        self.freeStructure = False
        self.endTurnEffect = None

    def assignPersonality(self, personality):
        self.personality = personality

    def assignWonders(self, wonder):
        self.wonders = wonder
        beginRes = wonder.beginResource
        self.resource[beginRes.resource] += beginRes.amount

    def assignLeftRight(self, leftPlayer, rightPlayer):
        self.left = leftPlayer
        self.right = rightPlayer

    def printPlayer(self):
        print(self.__dict__)
        self.wonders.printWonder()
        for card in self.card:
            print(card.name, end=" ")

    def assignHand(self, hand):
        self.hand = hand

    def cardExist(self, name):
        for singleCard in self.card:
            if singleCard.name == name:
                return True
        return False

    def resourceExist(self, resourceType):  # Use in case of neighbor Player
        if self.resource[resourceType] > 0:
            return True
        for card in self.choosecard:
            for choose in card.getResource["resource"]:
                if choose["type"] == resourceType:
                    return True
        return False

    def checkLeftRight(self, amount, resourceType):
        leftPrice = self.westTradePrices[resourceType]
        rightPrice = self.eastTradePrices[resourceType]
        minPrice = 10000000
        side = "M"
        if self.coin >= leftPrice * amount and self.left.resourceExist(resourceType):
            minPrice = leftPrice * amount
            side = "L"
        if self.coin >= rightPrice * amount and self.right.resourceExist(resourceType):
            if minPrice > rightPrice * amount:
                minPrice = rightPrice * amount
                side = "R"
        if side == "M":
            return -1, side
        return minPrice, side

    def addiResComp(self, targetArr, curResArr):
        # print("BEFORE")
        # for i in targetArr:
        #    i.printResource()
        for res in curResArr:
            name = res.resource
            # print("Name" + name)
            for tar in targetArr:
                if name == tar.resource:
                    tar.amount -= res.amount
        targetArr = [i for i in targetArr if i.amount > 0]
        # print("AFTER")
        # for i in targetArr:
        #   i.printResource()
        return targetArr

    def BFS(self, targetArray, resourceArray):
        queue = []
        minLeft = 10000000
        minRight = 10000000
        queue.append(ResourceBFS([], resourceArray))
        while queue:
            left = 0
            right = 0
            price = -1
            side = "M"
            # print(queue[0])
            qFront = queue[0]
            curRes = qFront.accuArr
            # print(curRes)
            remainRes = qFront.remainArr
            # print("REMAINRES")
            # print(remainRes)
            remainArr = self.addiResComp(targetArray[:], curRes[:])
            for remain in remainArr:
                price, side = self.checkLeftRight(remain.amount, remain.resource)
                if price == -1:
                    break
                if side == "L":
                    left += price
                elif side == "R":
                    right += price
            if price != -1 and left + right < minLeft + minRight:
                minLeft = left
                minRight = right
            queue.pop(0)
            if remainRes:
                resChooseCard = remainRes[0]
                for res in resChooseCard.getResource["resource"]:
                    # print("RES")
                    # print(res)
                    selectRes = mainHelper.resBuild(res)
                    newResAccu = curRes[:]
                    newResAccu.append(selectRes)
                    # print('NewResAccu : {}'.format(newResAccu))
                    newRemRes = remainRes[1:]
                    if newRemRes is None:
                        newRemRes = []
                    queue.append(ResourceBFS(newResAccu, newRemRes))
        return minLeft, minRight

    def playable(self, card):
        # if isinstance(card,Stage):
        #    print("IN STAGE")
        # print(card.payResource)
        # print("--------------")
        # print(card.payResource)
        payRes = card.payResource
        if payRes["type"] == "choose":
            if self.cardExist(payRes["resource"][0]["name"]):
                return 0, 0
            else:
                payRes = payRes["resource"][1]
                # print("NEW PAYRES-----")
                # print(payRes)
        if payRes["type"] == "none":
            return 0, 0
        elif payRes["type"] == "coin":
            if self.coin >= payRes["amount"]:
                return 0, 0
            else:
                return -1, -1
        else:
            missResource = {}
            if payRes["type"] == "mixed":
                for res in payRes["resource"]:
                    if self.resource[res["type"]] < res["amount"]:
                        missResource[res["type"]] = res["amount"] - self.resource[res["type"]]
            else:
                res = payRes
                if self.resource[res["type"]] < res["amount"]:
                    missResource[res["type"]] = res["amount"] - self.resource[res["type"]]
            if len(missResource) == 0:
                return 0, 0
            missResource = dict(sorted(missResource.items(), key=operator.itemgetter(1), reverse=True))
            # print("oldMissResource")
            # print(missResource)
            missArr = []
            for name, amount in missResource.items():
                missArr.append(Resource(name, amount))
            left, right = self.BFS(missArr, self.choosecard + self.chooseStage)
            if self.coin >= left + right:
                return left, right
            else:
                return -1, -1

    def findCardFromHand(self, cardName):
        for card in self.hand:
            if card.name == cardName:
                return card
        return None

    def activateEffect(self, effect):
        if effect == "freeStructure":
            self.freeStructure = True
            self.endAgeEffect.append(effect)
        elif effect == "playSeventhCard":
            self.endAgeEffect.append(effect)
        elif effect == "buildDiscarded":
            self.endTurnEffect = effect
        elif effect == "sideTradingRaws":
            self.activateEffect("eastTradingRaws")
            self.activateEffect("westTradingRaws")
        elif effect == "eastTradingRaws":
            self.eastTradePrices["wood"] = 1
            self.eastTradePrices["clay"] = 1
            self.eastTradePrices["ore"] = 1
            self.eastTradePrices["stone"] = 1
        elif effect == "westTradingRaws":
            self.westTradePrices["wood"] = 1
            self.westTradePrices["clay"] = 1
            self.westTradePrices["ore"] = 1
            self.westTradePrices["stone"] = 1
        elif effect == "sideManuPosts":
            self.eastTradePrices["papyrus"] = 1
            self.eastTradePrices["glass"] = 1
            self.eastTradePrices["loom"] = 1
            self.westTradePrices["papyrus"] = 1
            self.westTradePrices["glass"] = 1
            self.westTradePrices["loom"] = 1
        elif effect == "threeBrownOneCoin":
            self.coin += self.left.color["brown"] + self.color["brown"] + self.right.color["brown"]
        elif effect == "brownOneCoinOneVP":
            self.coin += self.color["brown"]
            self.endGameEffect.append(effect)
        elif effect == "yellowOneCoinOneVP":
            self.coin += self.color["yellow"]
            self.endGameEffect.append(effect)
        elif effect == "stageThreeCoinOneVP":
            self.coin += self.wonders.stage * 3
            self.endGameEffect.append(effect)
        elif effect == "greyTwoCoinTwoVP":
            self.coin += self.color["grey"] * 2
            self.endGameEffect.append(effect)
        else:  # effect == "sideBrownOneVP","sideGreyTwoVP","sideYellowOneVP","sideGreenOneVP","sideRedOneVP",
            # "sideDefeatOneVP","brownGreyPurpleOneVP", "brownOneCoinOneVP", "yellowOneCoinOneVP", "stageThreeCoinOneVP",
            # "greyTwoCoinTwoVP","copyPurpleNeighbor"
            self.endGameEffect.append(effect)

    def VPFromSide(self, col, mult):
        return (self.left.color[col] + self.right.color[col]) * mult

    def VPFromEffect(self, effect):
        if effect == "sideBrownOneVP":
            return self.VPFromSide("brown", 1)
        elif effect == "sideGreyTwoVP":
            return self.VPFromSide("grey", 2)
        elif effect == "sideYellowOneVP":
            return self.VPFromSide("yellow", 1)
        elif effect == "sideGreenOneVP":
            return self.VPFromSide("green", 1)
        elif effect == "sideRedOneVP":
            return self.VPFromSide("red", 1)
        elif effect == "sideDefeatOneVP":
            return self.left.warLoseVP + self.right.warLoseVP
        elif effect == "brownGreyPurpleOneVP":
            return self.color["brown"] + self.color["grey"] + self.color["purple"]
        elif effect == "brownOneCoinOneVP":
            return self.color["brown"]
        elif effect == "yellowOneCoinOneVP":
            return self.color["yellow"]
        elif effect == "stageThreeCoinOneVP":
            return self.wonders.stage
        elif effect == "greyTwoCoinTwoVP":
            return self.color["grey"] * 2
        else:
            return 0

    def scienceVP(self, chooseScience):
        maxScience = 0
        compass = self.resource["compass"]
        wheel = self.resource["wheel"]
        tablet = self.resource["tablet"]
        if chooseScience == 0:
            return compass**2 + wheel**2 + tablet**2 + min(compass, wheel, tablet) * 7
        for addCom in range(0, chooseScience):
            for addWheel in range(0, chooseScience - addCom):
                addTab = chooseScience - addCom - addWheel
                compass = addCom + self.resource["compass"]
                wheel = addWheel + self.resource["wheel"]
                tablet = addTab + self.resource["tablet"]
                points = compass**2 + wheel**2 + tablet**2 + min(compass, wheel, tablet) * 7
                if points > maxScience:
                    maxScience = points

        return maxScience

    def endGameCal(self):
        extraPoint = 0
        # print("Player" + str(self.player))
        # print("Before" + str(self.VP))
        # military
        extraPoint += self.warVP - self.warLoseVP
        # print("War : " + str(self.warVP - self.warLoseVP))
        # coin : 3coins -> 1VP
        extraPoint += int(self.coin / 3)
        # print("Coin : " + str(int(self.coin/3)))
        # wonders VP are automatically added when wonders are played
        # effect cards activation
        copyPurple = False
        for effect in self.endGameEffect:
            if effect == "copyPurpleNeighbor":
                copyPurple = True
            else:
                extraPoint += self.VPFromEffect(effect)
        maxCopy = 0
        scienceNeighbor = False
        if copyPurple:
            purple = []
            for card in self.left.card:
                if card.color == "purple":
                    purple.append(card)
            for card in self.right.card:
                if card.color == "purple":
                    purple.append(card)
            maxPt = 0
            for card in purple:
                if card.name == "scientists guild":
                    scienceNeighbor = True
                else:
                    if maxPt < self.VPFromEffect(card.getResource["effect"]):
                        maxPt = self.VPFromEffect(card.getResource["effect"])
            maxCopy = maxPt
        # science points
        chooseScience = 0
        for card in self.choosecard:
            if card.getResource["type"] == "choose" and card.getResource["resource"][0]["type"] == "compass":
                chooseScience += 1
        for i in range(0, self.wonders.stage):
            if (
                self.wonders.step[i + 1].getResource["type"] == "choose"
                and self.wonders.step[i + 1].getResource["resource"][0]["type"] == "compass"
            ):
                chooseScience += 1
        maxScience = self.scienceVP(chooseScience)
        if scienceNeighbor:
            copyScience = self.scienceVP(chooseScience + 1)
            if maxScience + maxCopy < copyScience:
                maxScience = copyScience
        # print("Science : " + str(maxScience))
        extraPoint += maxScience
        return self.VP + extraPoint

    def deleteCardFromHand(self, card):
        if any(cardExist for cardExist in self.hand if cardExist.name == card.name):
            self.hand.remove(card)
            return 1
        else:
            return -1

    def addedCardSys(self, cardGetResource, selectedCard):
        if cardGetResource["type"] == "choose":
            if isinstance(selectedCard, Stage):
                self.chooseStage.append(selectedCard)
            else:
                self.choosecard.append(selectedCard)
        elif cardGetResource["type"] == "VP":
            self.VP += cardGetResource["amount"]
        elif cardGetResource["type"] == "coin":
            self.coin += cardGetResource["amount"]
        elif cardGetResource["type"] != "effect":
            self.resource[cardGetResource["type"]] += cardGetResource["amount"]
        else:
            self.activateEffect(cardGetResource["effect"])
            self.lastPlayEffect = cardGetResource["effect"]
            return

    def playChosenCard(self, selectedCard):
        self.lastPlayEffect = None
        leftPrice = selectedCard[1]
        rightPrice = selectedCard[2]
        action = selectedCard[3]
        stage = None
        if len(selectedCard) == 5:
            stage = selectedCard[4]
        selectedCard = selectedCard[0]
        if action == -1:
            # print("SELECT")
            # if isinstance(selectedCard,Card):
            #     print(selectedCard.name)
            # else:
            #     print(selectedCard.stage)
            status = self.deleteCardFromHand(selectedCard)
            if not status:
                print("ERROR STATUS")
            self.coin += 3
            return selectedCard, action
        elif action == 1:
            self.freeStructure = False
        if isinstance(selectedCard, Card):
            # print(selectedCard.name)
            status = self.deleteCardFromHand(selectedCard)
            self.card.append(selectedCard)
            self.color[selectedCard.color] += 1
            self.lastPlayColor = selectedCard.color
        elif action == 2:
            status = self.deleteCardFromHand(selectedCard)
            self.wonders.stage += 1
            # print(self.wonders.name)
            # print(self.wonders.step[self.wonders.stage].printCard())
        if selectedCard.getResource["type"] == "mixed":
            for resource in selectedCard.getResource["resource"]:
                if resource["type"] == "mixed":
                    for innerRes in resource["resource"]:
                        self.addedCardSys(innerRes, selectedCard)
                else:
                    # print(resource["type"])
                    self.addedCardSys(resource, selectedCard)
        elif selectedCard.getResource["type"] == "choose":
            if isinstance(stage, Stage):
                self.chooseStage.append(stage)
            else:
                self.choosecard.append(selectedCard)
        else:
            self.addedCardSys(selectedCard.getResource, selectedCard)
        return selectedCard, action

    def playChosenCardFake(self, selectedCard):
        self.lastPlayEffect = None
        leftPrice = selectedCard[1]
        rightPrice = selectedCard[2]
        action = selectedCard[3]
        stage = None
        if len(selectedCard) == 5:
            stage = selectedCard[4]
        selectedCard = selectedCard[0]
        if action == -1:
            # print("SELECT")
            # if isinstance(selectedCard,Card):
            #     print(selectedCard.name)
            # else:
            #     print(selectedCard.stage)
            # self.deleteCardFromHand(selectedCard)
            self.coin += 3
            return selectedCard, action
        elif action == 1:
            self.freeStructure = False
        if isinstance(selectedCard, Card):
            # print(selectedCard.name)
            # self.deleteCardFromHand(selectedCard)
            self.card.append(selectedCard)
            self.color[selectedCard.color] += 1
            self.lastPlayColor = selectedCard.color
        elif action == 2:
            # self.deleteCardFromHand(stageCard)
            self.wonders.stage += 1
            # print(self.wonders.name)
            # print(self.wonders.step[self.wonders.stage].printCard())
        if selectedCard.getResource["type"] == "mixed":
            for resource in selectedCard.getResource["resource"]:
                if resource["type"] == "mixed":
                    for innerRes in resource["resource"]:
                        self.addedCardSys(innerRes, selectedCard)
                else:
                    # print(resource["type"])
                    self.addedCardSys(resource, selectedCard)
        elif selectedCard.getResource["type"] == "choose":
            if isinstance(selectedCard, Stage):
                self.chooseStage.append(stage)
            else:
                self.choosecard.append(selectedCard)
        else:
            self.addedCardSys(selectedCard.getResource, selectedCard)
        return selectedCard, action

    def playFromEffect(self, cardList, effect, age):  # playSeventhCard or buildDiscarded
        if effect == "playSeventhCard":
            card = cardList[0]
            choices = []
            choices.append([card, 0, 0, -1])
            left, right = self.playable(card)
            if left != -1 and right != -1:
                choices.append([card, left, right, 0])
            steps = self.wonders.step
            # print("LEN STEPS : {}".format(len(steps)))
            existedStage = self.wonders.stage + 1
            # print(type(steps[existedStage]))
            if existedStage < len(steps):
                left, right = self.playable(steps[existedStage])
                if left != -1 and right != -1:
                    for card in self.hand:
                        choices.append([card, left, right, 2, steps[existedStage]])
                # print("APPENDED STAGES")
            persona = self.personality
            selectedCard = choices[persona.make_choice(self=persona, player=self, age=age, options=choices)]
            return self.playChosenCard(selectedCard)
        elif effect == "buildDiscarded":
            choices = []
            for card in cardList:
                choices.append([card, 0, 0, 0])
            persona = self.personality
            selectedCard = choices[persona.make_choice(self=persona, player=self, age=age, options=choices)]
            self.hand.append(selectedCard[0])
            return self.playChosenCard(selectedCard)
        else:
            print("something wrong")
            exit(-1)

    def playCard(self, age):
        self.lastPlayEffect = None
        self.endTurnEffect = None
        choices = []
        for card in self.hand:
            choices.append([card, 0, 0, -1])  # -1 for discard card
        for card in self.hand:
            left, right = self.playable(card)
            if left != -1 and right != -1:
                choices.append([card, left, right, 0])  # card,leftPrice,rightPrice,0 for not using free effect
        if self.freeStructure == True:
            for card in self.hand:
                choices.append([card, 0, 0, 1])  # card,leftPrice,rightPrice,1 for using free effect
        steps = self.wonders.step
        existedStage = self.wonders.stage + 1
        if existedStage < len(steps):
            left, right = self.playable(steps[existedStage])
            if left != -1 and right != -1:
                for card in self.hand:
                    choices.append([card, left, right, 2, steps[existedStage]])  # Append Stage
        persona = self.personality
        selectedCard = choices[persona.make_choice(self=persona, player=self, age=age, options=choices)]
        print("SELECT", selectedCard)
        return self.playChosenCard(selectedCard)
