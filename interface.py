import numpy as np
import cv2
from matplotlib import pyplot as plt
import matplotlib.image as im

# img = cv2.imread('example.png , 0)
# plt.imshow( cv2.cvtColor( img, cv2.COLOR_BGR2RGB))  converts opencv read img to plot

img = im.imread('example.png', 0) #read img as original with matplotlib
plt.imshow(img)
plt.axis('off')  # hides x,y axises
plt.show()