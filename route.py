import random
import Rockhold

class Route:
    def __init__(self, holdList, grade, imageCoordinates, color=None):
        self.holdList = holdList
        self.grade = grade
        self.color = color
        self.route = []
        self.imageCoordinates = imageCoordinates
        self.handHoldList = [hold for hold in holdList if hold.type != "Foothold"]
        self.footHoldList = [hold for hold in holdList if hold.type == "Foothold"]

    def findStart(self, bottomThreshold):
        eligibleStarts = [hold for hold in self.holdList if hold.center[1] >= bottomThreshold and hold.type != "Foothold"]
        if len(eligibleStarts) == 0:
            return None
        return random.choice(eligibleStarts)
    
    def generateRoute(self):
        start = self.findStart(1200)
        if start is None:
            return None
        start.drawColor = (0, 0, 255)
        self.route.append(start)
        currentHold = start
        while currentHold.center[1] > self.imageCoordinates[1] / 6:
            nextHold = self.findNextHold(currentHold, 200)
            if nextHold is None:
                return None
            self.route.append(nextHold)
            currentHold = nextHold
        currentHold.drawColor = (0, 255, 255)
        self.generateFeet()

# int(random.triangular(-distanceThreshold, distanceThreshold, 0))

    def findNextHold(self, currentHold, distanceThreshold):
        nextPoint = (currentHold.center[0]  + int(random.triangular(-distanceThreshold, distanceThreshold, 0)), currentHold.center[1] - int(random.triangular(0, distanceThreshold * 2, distanceThreshold)))

        eligibleHolds = [hold for hold in self.handHoldList if hold.distance(currentHold) <= distanceThreshold * 1.5 and hold not in self.route]
        
        weights = [hold.point_distance(nextPoint) + (2 * (self.imageCoordinates[0] - hold.center[1])) for hold in eligibleHolds]
        if len(eligibleHolds) == 0:
            return None
        return random.choices(eligibleHolds, weights=weights, k=1)[0]
    

    def draw(self,scene):
        for hold in self.route:
            scene = hold.draw(scene)
        return scene
    

    def generateFeet(self):
        for handHold in self.handHoldList:
            areaToCheck = (handHold.center[0], handHold.center[1] + 100)
            alreadyHasFoot = False
            # alreadyHasFoot = len([hold for hold in self.route if hold.point_distance(areaToCheck) < 50]) > 0
            if not alreadyHasFoot:
                eligibleHolds = [hold for hold in self.footHoldList if hold.point_distance(areaToCheck) < 50]
                weights = [hold.center[1] for hold in eligibleHolds]
                if len(eligibleHolds) > 0:
                    foot = random.choices(eligibleHolds, weights=weights, k=1)[0]
                    foot.drawColor = (255, 0, 0)
                    self.route.append(foot)