class Stage:
    def __init__(self, number, payResource, getResource):
        self.stage = number
        self.payResource = payResource
        self.getResource = getResource
    def printCard(self):
        print("step:" + str(self.stage))
        print(self.payResource)
        print(self.getResource)