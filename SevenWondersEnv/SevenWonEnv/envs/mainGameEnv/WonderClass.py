from SevenWonEnv.envs.mainGameEnv.stageClass import Stage


class Wonder:
    def __init__(self, name, side, beginResource, **kwargs):
        self.name = name
        self.side = side
        self.stage = 0
        self.step = {}

        self.beginResource = beginResource
        steps = kwargs
        # print(len(steps))
        for level in steps:
            # print("build level : {}".format(level))
            levelNum = int(level[6])
            self.step[levelNum] = Stage(levelNum, kwargs[level]["payResource"], kwargs[level]["getResource"])

    def printWonder(self):
        print(self.__dict__)
        self.beginResource.printResource()
