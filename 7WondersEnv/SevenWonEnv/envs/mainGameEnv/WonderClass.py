from SevenWonEnv.envs.mainGameEnv.resourceClass import Resource
from SevenWonEnv.envs.mainGameEnv.stageClass import Stage
class Wonder:
    def __init__(self, name,side,resourceIni,resourceAmount, **kwargs):
        self.name = name
        self.side = side
        self.stage = 0
        self.step = {}

        self.beginResource = Resource(resourceIni,resourceAmount)
        steps = kwargs
        #print(len(steps))
        for level in steps:
            #print("build level : {}".format(level))
            levelNum = int(level[6])
            self.step[levelNum] = Stage(levelNum,kwargs[level]["payResource"],kwargs[level]["getResource"])
    def printWonder(self):
        print(self.__dict__)
        self.beginResource.printResource()

