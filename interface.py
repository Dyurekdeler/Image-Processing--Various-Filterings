from skimage import data,io
from skimage.viewer import ImageViewer

#load img
img = io.imread('example.png')

#display
viewer = ImageViewer(img)
viewer.show()