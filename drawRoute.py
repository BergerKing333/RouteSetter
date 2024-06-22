import cv2
import numpy as np

colors = [(0, 0, 0), (252, 107, 3), (0, 95, 227), (5, 181, 26), (150, 6, 212)]

def drawImg(img, holdArray, holdCoordinates):
    for i in range(len(holdArray)):
        if holdArray[i] != 0:
            cv2.circle(img, holdCoordinates[i], 5, colors[holdArray[i]], -1)
    return img