#!/usr/local/bin/python3
import cv2 as cv
import numpy as np

# Load the image and convert to HSV colourspace
image = cv.imread("acnepaint.png")
org = cv.imread("acnepaint.png")
hsv=cv.cvtColor(image,cv.COLOR_BGR2HSV)

# Define lower and uppper limits of red
brown_lo=np.array([20,20,50])
brown_hi=np.array([250,250,255])

# Mask image to only select browns
mask=cv.inRange(hsv,brown_lo,brown_hi)

# Change image to beige where red is found
image[mask>0]=(175,201,237)

cv.imshow("result",image)

cv.waitKey(0)