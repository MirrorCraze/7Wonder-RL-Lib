class Card:
    def __init__(self, name,color, payResource, getResource):
        self.name = name
        self.color = color
        self.payResource = payResource
        self.getResource = getResource
    def printCard(self):
        print("name:" + self.name)
        print(self.payResource)
        print(self.getResource)