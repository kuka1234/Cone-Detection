import random
from enum import Enum
import numpy as np
import coneDetector

pathToImg = r"C:\Users\shree\PycharmProjects\coneDetection\detection\Images\350B0489.jpg"
img = coneDetector.loadImage(pathToImg)
if type(img) is None:
    raise Exception("Path not valid")


# coneDetector.colorPicker(img)

coneDetector.debugMode = False
coneDetector.timeMode = False

class Colors(Enum):
    # yellow = (13,14, True)
    # blue = (115,3, False)
    orange = (9,10,False)

def simplePipeline(img):
    finalImg = np.zeros_like(img)
    for col in Colors:
        color,colorRange,black = col.value
        colorImg = coneDetector.getColour(img, color,
                                          saturation=60, colorRange=colorRange)  # val=2 extracts orange cones, val=(108,120) extracts blue cones,val=22 for yellow.
        grayImg = coneDetector.getBlackorWhite(img, black)
        coneDetector.getContours(coneDetector.postProcess(colorImg), maxSize=250, maskedOutput=finalImg)
        coneDetector.getContours(coneDetector.postProcess(grayImg), maxSize=120, maskedOutput=finalImg)
        contours, cones = coneDetector.getContours(coneDetector.postProcess(finalImg), maxSize=250,maskedOutput=finalImg)
        coneDetector.drawRectForContours(img, contours, (0, 255, 0))
        coneDetector.showImg(img)

def complexPipeline(img):
    finalContours = []
    for col in Colors:
        color,colorRange,black = col.value
        print(f"Extract {col.name} coloured cones. Has black stripe: {black}")
        rectCol = (0,random.randint(0,255), random.randint(0,255))
        finalImg = np.zeros_like(img)
        grayImg = coneDetector.getBlackorWhite(img, black)
        contours, cones = coneDetector.getContours(coneDetector.postProcess(grayImg), maxSize=120)
        maskedImg = coneDetector.createMasked(img, contours, black)
        colorImg = coneDetector.getColour(maskedImg, color,
                                          saturation=60, colorRange=colorRange)

        contours,cones = coneDetector.getContours(coneDetector.postProcess(colorImg), minArea=50, maxSize=300,maskedOutput=finalImg)
        maskedImg = coneDetector.createMasked(img, contours, True)
        grayImg = coneDetector.getBlackorWhite(maskedImg, True)
        coneDetector.getContours(coneDetector.postProcess(grayImg),minArea=50, maskedOutput=finalImg)
        contours, cones = coneDetector.getContours(coneDetector.postProcess(finalImg), minArea=75)
        finalContours.append((contours, rectCol))

    for contour,rectCol in finalContours:
        coneDetector.drawRectForContours(img, contour, rectCol)
    coneDetector.showImg(img)

simplePipeline(img)
# complexPipeline(img)


