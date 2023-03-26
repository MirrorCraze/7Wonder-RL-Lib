class Resource:
    def __init__(self, resource, amount):
        self.resource = resource
        self.amount = amount

    def printResource(self):
        print(self.__dict__)
