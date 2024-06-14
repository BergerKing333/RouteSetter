import cv2
import numpy as np
import math as Math

class Rockhold:
    def __init__(self, type, startX, startY, endX, endY):
        self.type = type
        self.startX = startX
        self.startY = startY
        self.endX = endX
        self.endY = endY
        self.width = abs(self.endX - self.startX)
        self.height = abs(self.endY - self.startY)
        self.center = (int(self.startX + self.width / 2), int(self.startY + self.height / 2))
        self.drawColor = (0, 255, 0)

    def draw(self, scene):
        cv2.circle(scene, self.center, 10, self.drawColor, -1)
        return scene
    
    def point_distance(self, point):
        return Math.sqrt((self.center[0] - point[0]) ** 2 + (self.center[1] - point[1]) ** 2)

    def distance(self, other):
        return Math.sqrt((self.center[0] - other.center[0]) ** 2 + (self.center[1] - other.center[1]) ** 2)