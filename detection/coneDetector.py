import copy
import time

import cv2
import numpy as np
import random

debugMode = False
timeMode = False

class Accumulator:
    '''
    Counts total amount of time for image processing
    '''
    total = 0
    @classmethod
    def add(cls, x):
        cls.total += x
        return cls.total

def showImgAutomatic(func):
    '''
    Wrapper to show image after every change. Useful for debugging.
    '''
    if debugMode:
        print(f"Debug Mode: Output of the {func.__name__} function")
    def wrapper(*args, **kwargs):
        if debugMode:
            image = func(*args,**kwargs)
            showImg(image, func.__name__)
            return image
        else:
            image = func(*args,**kwargs)
            return image
    return wrapper

def timeFunc(func):
    '''
    Wrapper to measure time of each function
    '''
    def wrapper(*args, **kwargs):
        if timeMode:
            start = time.time()
            image = func(*args,**kwargs)
            end = time.time()
            total = Accumulator.add(end-start)
            print(f"Time Mode: {func.__name__} function took : {round(end-start,4)} seconds. \n "
                  f"Total time so far: {round(total,4)} seconds")
            return image
        else:
            image = func(*args,**kwargs)
            return image
    return wrapper

@timeFunc
def loadImage(path):
    img = cv2.imread(path)
    return img

@timeFunc
def showImg(img, name="Image") -> None:
    '''
    Shows image and waits until key pressed.
    :param img: image to be shown
    '''
    cv2.imshow(name,img)
    cv2.waitKey(0)

@showImgAutomatic
@timeFunc
def getColour(img, color, saturation=10, colorRange=2):
    '''
    Extracts color from image
    :param img: image to be extracted from
    :param color: mid of the color range to be extracted
    :return: newImage
    '''
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    low = np.array([max(color - colorRange, 0), saturation, 0])
    high = np.array([min(color + colorRange, 180), 255, 255])
    m = cv2.inRange(imgHSV, low, high)
    newImage = cv2.bitwise_and(img, img, mask=m)
    return newImage

@showImgAutomatic
@timeFunc
def getBlackorWhite(img,black):
    '''
    Extracts black or white pixels from image
    :param img: image to be extracted from
    :param black: extract black if True else extracts False
    :return: mask of the required color
    '''
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    if black:
        low = np.array([0,0,0])
        high = np.array([180, 60, 245])
    else:
        low = np.array([0, 210, 0])
        high = np.array([255, 255, 255])
    m = cv2.inRange(imgHSV, low, high)
    return m

@showImgAutomatic
@timeFunc
def postProcess(img):
    '''
    Applies post processing effects
    :param img: image to processed
    :return: post processed image
    '''
    blur = cv2.GaussianBlur(img, (7, 7), 2)
    canny = cv2.Canny(blur, 75, 75)
    dilated = cv2.dilate(canny, (2, 2), iterations=1)
    blur = cv2.GaussianBlur(dilated, (3, 3), 1)
    return blur

@timeFunc
def getContours(img, minArea=100, maxSize=500, maskedOutput = [], thickness=-1):
    '''
    Extracts cones of images, works best when background is black and only objects are colored
    :param img: image to get contours of
    :param maskedOutput: option to draw counter over image
    :return: mask of the contours, nested list all pixels within the contour respective to the contour
    '''
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    finalContours = []
    cone_pixels = []
    for ind, contour in enumerate(contours):
        dim = cv2.boundingRect(contour)
        if cv2.contourArea(contour)> minArea and dim[2] < maxSize and dim[3]<maxSize:
            mask = np.zeros_like(img)
            cv2.drawContours(mask, [contour], -1, 255,thickness=-1)
            if len(maskedOutput) != 0:
                cv2.drawContours(maskedOutput, [contour], -1, 255,thickness=thickness)
            pts = np.where(mask == 255)
            cur_cone = []
            for i in range(len(pts[0])):
                cur_cone.append([pts[0][i], pts[1][i]])
            cone_pixels.append([cur_cone])
            finalContours.append(contour)
    return finalContours, cone_pixels

@showImgAutomatic
@timeFunc
def colorCones(img,cones):
    '''
    Colors each cone/contourArea a different color to clearly see areas highlighted
    :param img: base image to be coloured
    :param cones: nested list of areas containing pixels
    :return: coloured image
    '''
    img = copy.deepcopy(img)
    for ind, cone in enumerate(cones):
        randColor = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
        for pixel in cone[0]:
            img[pixel[0], pixel[1]] = randColor
    return img

@showImgAutomatic
@timeFunc
def drawRectForContours(img, contours, color=(0,0,255)):
    '''
    Draws rectangular boxes around contours
    :param img: image to be drawn on
    :param contours: contours to be used
    :param color: colour of the rectangles
    :return: output image
    '''
    for contour in contours:
        dim = cv2.boundingRect(contour)
        cv2.rectangle(img, (dim[0], dim[1]), (dim[0] + dim[2], dim[1] + dim[3]),color, 2)
    return img

@showImgAutomatic
@timeFunc
def createMasked(img, contours,black, dilation = 20):
    '''
    Only keeps part of image inside the boundary of the contours
    :param img: image to be extracted from
    :param contours: used for the bounding boxes
    :param black: If true will turn not needed area white
    :param dilation: The extra margin to be included around the bounding boxes
    :return: masked image
    '''
    mask = np.zeros_like(img)
    if black:
        for row in mask:
            for ind,item in enumerate(row):
                row[ind] = [255,255,255]

    for contour in contours:
        dim = cv2.boundingRect(contour)
        if cv2.contourArea(contour) > 100 and dim[2] < 500 and dim[3] < 500:
            for row in range(max(0, dim[1] - dilation), min(len(mask)-1,dilation+dim[1] + dim[3])):
                if black:
                    mask[row][dim[0]:dim[0]+dim[2]] = [0,0,0]
                else:
                    mask[row][dim[0]:dim[0]+dim[2]] = [255,255,255]

    if black:
        img = cv2.bitwise_or(img,mask)
    else:
        img = cv2.bitwise_and(img,mask)
    return img

def colorPicker(img):
    '''
    Used to get color and range values for the color of the cone you want
    :param img: image to be sampled from
    '''
    scopeFunc = getColour
    def onChangeColor(val):
        imageCopy = img.copy()

        imageCopy = scopeFunc(imageCopy, val, saturation=60, colorRange=cv2.getTrackbarPos('Range', "Image"))
        cv2.imshow("Image", imageCopy)

    def onChangeRange(val):
        imageCopy = img.copy()

        imageCopy = scopeFunc(imageCopy, cv2.getTrackbarPos('Color', "Image"), saturation=60, colorRange=val)
        cv2.imshow("Image", imageCopy)

    cv2.imshow("Image", img)
    cv2.createTrackbar('Color', "Image", 0, 180, onChangeColor)
    cv2.createTrackbar('Range', "Image", 2, 50, onChangeRange)
    cv2.waitKey(0)