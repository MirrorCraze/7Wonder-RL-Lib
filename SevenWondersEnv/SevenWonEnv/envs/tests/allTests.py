from SevenWonEnv.envs.mainGameEnv.mainHelper import (
    filterPlayer,
    buildCard,
    rotateHand,
    battle,
)
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
from SevenWonEnv.envs.mainGameEnv.cardClass import Card
from SevenWonEnv.envs.SevenWonderEnv import SevenWonderEnv

import gymnasium as gym
import unittest
from unittest import mock

# Unit Testing


class Test(unittest.TestCase):
    def test_AddNoPlayer(self):
        with mock.patch.object(SevenWonderEnv, "__init__", lambda sel, player: None):
            env = SevenWonderEnv(player=4)
            env.unAssignedPersonality = 4
            with self.assertRaises(SystemExit) as cm:
                env.step(0)
            exc = cm.exception
            assert exc.code == "Program Stopped : Some Players do not have personality."

    def test_AddTooManyPlayer(self):
        with mock.patch.object(SevenWonderEnv, "__init__", lambda sel, player: None):
            env = SevenWonderEnv(player=4)
            env.unAssignedPersonality = 0
            env.player = 4
            env.playerList = []
            with self.assertRaises(SystemExit) as cm:
                env.setPersonality([Human])
            exc = cm.exception
            assert exc.code == "Program Stopped : Add too many players' personality"

    def test_GetCardAge(self):
        with mock.patch.object(SevenWonderEnv, "__init__", lambda sel, player: None):
            env = SevenWonderEnv(player=3)
            cardList = {
                "age_1": {
                    "3players": {
                        "brown": [
                            {
                                "name": "lumber yard",
                                "payResource": {"type": "none"},
                                "getResource": {"type": "wood", "amount": 1},
                            }
                        ]
                    }
                }
            }
            result = env.getCardAge("age_1", 3, cardList)
            assert result[0].name == "lumber yard"
            assert result[0].color == "brown"
            assert result[0].getResource == {"type": "wood", "amount": 1}
            assert result[0].payResource == {"type": "none"}

    def testBattle(self):
        with mock.patch.object(Player, "__init__", lambda se, playNum, totNum, person: None):
            p1 = Player(0, 4, Human)
            p2 = Player(0, 4, Human)
            p1.resource = {}
            p2.resource = {}
            p1.resource["shield"] = 0
            p2.resource["shield"] = 1
            p1.warVP = 0
            p2.warVP = 0
            battle(p1, p2, age=1)
            assert p1.warVP == -1
            assert p2.warVP == 1
            p1.warVP = 0
            p2.warVP = 0
            battle(p1, p2, age=2)
            assert p1.warVP == -1
            assert p2.warVP == 3
            p1.resource["shield"] = 1
            battle(p1, p2, age=3)
            assert p1.warVP == -1
            assert p2.warVP == 3

    def testAssignWonders(self):
        with mock.patch.object(Wonder, "__init__", lambda se, name, side, begin: None):
            won = Wonder("Rhodes", "A", Resource("wood", 2))
            won.beginResource = Resource("wood", 2)
            player = Player(1, 4, Human)
            player.assignWonders(won)
            assert player.resource["wood"] == 2

    # Integration Testing
    def testInitGameWithAI(self):
        env = SevenWonderEnv(player=4)
        env.setPersonality([RuleBasedAI, RuleBasedAI, RuleBasedAI, RuleBasedAI])
        assert env.player == 4
        assert env.unAssignedPersonality == 0
        for i in range(1, 5):
            assert len(env.playerList[i].hand) == 7

    def testStepGameWithAI(self):
        env = SevenWonderEnv(player=4)
        env.setPersonality([RuleBasedAI, RuleBasedAI, RuleBasedAI, RuleBasedAI])
        assert env.player == 4
        assert env.unAssignedPersonality == 0
        for i in range(1, 5):
            assert len(env.playerList[i].hand) == 7
        rewardAll = env.step(0)
        assert len(rewardAll) == 4
        assert len(rewardAll[0]) == 4
        state = env.reset()
        assert len(state) == 70


if __name__ == "__main__":
    unittest.main()
