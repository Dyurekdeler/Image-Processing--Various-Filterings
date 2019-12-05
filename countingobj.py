from skimage import io, filters
from scipy import ndimage
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage import measure

image = io.imread('kedi.png')
im = image
im = rgb2gray(im)
val = filters.threshold_otsu(im)
drops = ndimage.binary_fill_holes(im < val)

labels = measure.label(drops)
print(labels.max())

plt.figure()
plt.imshow(drops, cmap='gray')

plt.figure()
plt.imshow(image)
plt.show()


