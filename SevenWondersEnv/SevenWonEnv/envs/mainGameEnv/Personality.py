import random
from sys import stdin
from copy import deepcopy
from SevenWonEnv.envs.mainGameEnv.stageClass import Stage
from SevenWonEnv.envs.mainGameEnv.cardClass import Card


class Personality:
    def __init__(self):
        pass

    def make_choice(self, player, age, options):
        pass


class DQNAI(Personality):  # placeholder
    def __init__(self):
        super().__init__()

    def make_choice(self, player, age, options):
        pass


class RuleBasedAI(Personality):
    def __init__(self):
        super().__init__()

    def make_choice(self, player, age, options):
        # return random.choice(range(len(options)))

        for choicesIndex in range(len(options)):
            if isinstance(options[choicesIndex][0], Stage):  # If stage is free, buy it. 50% to buy if it's not free.
                if options[choicesIndex][1] + options[choicesIndex][2] == 0 or random.randint(0, 1) % 2 == 0:
                    return choicesIndex
                else:
                    continue
        # options[choicesIndex[3]] is play code. If it's -1, it means discarded. 0 is play with paying, 1 is play with effect.
        if age < 3:
            posChoice = []
            nonDiscard = []
            for choicesIndex in range(len(options)):
                if options[choicesIndex][3] != -1:
                    nonDiscard.append(choicesIndex)
            nonDiscarded = [option for option in options if option[3] != -1]
            if not nonDiscarded:  # have only choice by discarding
                return random.choice(range(len(options)))
            for choicesIndex in range(
                len(options)
            ):  # Select Card that gives more than 1 resource. If there are multiple cards, select one randomly
                if type(options[choicesIndex][0]).__name__ == "Card":
                    if options[choicesIndex][0].getResource["type"] == "mixed" and options[choicesIndex][3] != -1:
                        posChoice.append(choicesIndex)
            if posChoice:
                return random.choice(posChoice)
            for choicesIndex in range(
                len(options)
            ):  # Select Card that can be selected between resource. If there are multiple cards, select one randomly
                if isinstance(options[choicesIndex][0], Card):
                    if options[choicesIndex][0].getResource["type"] == "choose" and options[choicesIndex][3] != -1:
                        posChoice.append(choicesIndex)
            if posChoice:
                return random.choice(posChoice)
            zeroRes = {key: value for (key, value) in player.resource.items() if value == 0 and key != "shield"}
            for choicesIndex in range(
                len(options)
            ):  # Select resource that does not have yet (0 resource) except military. If there are multiple cards, select one randomly
                if isinstance(options[choicesIndex][0], Card):
                    for res in zeroRes.keys():
                        if options[choicesIndex][0].getResource["type"] == res and options[choicesIndex][3] != -1:
                            posChoice.append(choicesIndex)
            if posChoice:
                return random.choice(posChoice)
            if not (
                player.resource["shield"] > player.left.resource["shield"]
                or player.resource["shield"] > player.right.resource["shield"]
            ):
                for choicesIndex in range(
                    len(options)
                ):  # Select military IF it makes player surpass neighbors in shield. If there are multiple cards, select one randomly
                    if isinstance(options[choicesIndex][0], Card):
                        if options[choicesIndex][0].getResource["type"] == "shield" and options[choicesIndex][3] != -1:
                            shieldPts = options[choicesIndex][0].getResource["amount"]
                            if (
                                player.resource["shield"] + shieldPts > player.left.resource["shield"]
                                or player.resource["shield"] + shieldPts > player.right.resource["shield"]
                            ):
                                posChoice.append(choicesIndex)
            if posChoice:
                return random.choice(posChoice)
            for choicesIndex in range(len(options)):  # Select science card. If there are multiple cards, select one.
                if isinstance(options[choicesIndex][0], Card):
                    if options[choicesIndex][0].color == "green" and options[choicesIndex][3] != -1:
                        posChoice.append(choicesIndex)

            if posChoice:
                return random.choice(posChoice)
            for choicesIndex in range(len(options)):  # Select VP (civil) card. If there are multiple cards, select one.
                if isinstance(options[choicesIndex][0], Card):
                    if options[choicesIndex][0].getResource["type"] == "VP" and options[choicesIndex][3] != -1:
                        if not posChoice:
                            posChoice.append(choicesIndex)
                        elif (
                            options[posChoice[0]][0].getResource["amount"]
                            < options[choicesIndex][0].getResource["amount"]
                        ):
                            posChoice = [choicesIndex]

            if posChoice:
                return random.choice(posChoice)
            # play random non-discarded choice
            return random.choice(nonDiscard)
        else:  # age 3. Simulate all plays, greedy by most points.
            basePoints = player.endGameCal()
            gainPoints = -1
            selected = []
            for choicesIndex in range(len(options)):
                afterPlayer = deepcopy(player)
                afterPlayer.hand = deepcopy(player.hand)
                # print("HAND")
                # print(len(afterPlayer.hand))
                # print(choicesIndex)
                # print(options[choicesIndex])
                afterPlayer.playChosenCardFake(options[choicesIndex])
                addPoints = afterPlayer.endGameCal() - basePoints
                if addPoints < 0:
                    print("WRONG")
                if addPoints > gainPoints:
                    selected = [choicesIndex]
                    gainPoints = addPoints
                elif addPoints == gainPoints:
                    selected.append(choicesIndex)
            if selected:
                return random.choice(selected)
            else:
                return random.choice(range(len(options)))


class Human(Personality):
    def __init__(self):
        super().__init__()

    def make_choice(self, player, age, options):
        return int(stdin.readline())


class RandomAI(Personality):
    def __init__(self):
        super().__init__()

    def make_choice(self, player, age, options):
        return random.choice(range(len(options)))
