import tkinter
from tkinter import filedialog
from tkinter import ttk
import skimage
from scipy.ndimage import gaussian_filter
from skimage import io; io.use_plugin('matplotlib')
from skimage import img_as_float, data
from skimage.exposure import exposure
from skimage.transform import rescale, resize, downscale_local_mean, swirl
import cv2
from skimage.filters import rank
from scipy import ndimage
from scipy.ndimage import generic_gradient_magnitude, rotate
from skimage.color import rgb2gray
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from skimage.morphology import disk

import numpy as np
from skimage.filters import unsharp_mask, sobel

class GUI():

    def __init__(self):
        self.root = tkinter.Tk()
        self.root.wm_title("Densy's Image World")
        self.activeCanvas = None
        self.activeFilename = None
        self.combobox = None
        self.createWidgets()
  #https://answers.opencv.org/question/97416/replace-a-range-of-colors-with-a-specific-color-in-python/
        tkinter.mainloop()

    def createWidgets(self):

        fig = Figure(figsize=(5, 4), dpi=100)
        a = fig.add_subplot(111)
        canvas = FigureCanvasTkAgg(fig, master=self.root)  # A tk.DrawingArea.
        canvas.draw()
        canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)

        quitbtn = tkinter.Button(master=self.root, text="Quit", command=self._quit)
        quitbtn.pack(side=tkinter.BOTTOM)

        browsebtn = tkinter.Button(self.root, text="Browse A File", command= lambda: self.browse(a, canvas, 'default'))
        browsebtn.pack()

        combo = ttk.Combobox(self.root,
                                    values=[
                                        "gray",
                                        "very_blurry",
                                        "sharp",
                                        "motion_blur",
                                        "contrast","contour","find_edges","lighten","darken","invert","gaussian",
                                        "percentile_mean","bilateral_mean",

                                    "crop","rescale","resize","downscaled","rotate","swirl",
                                    
                                    "intensity",

                                    "mean",

                                    "histogram",

                                    "erosion"])
        combo.current(0)

        combo.bind("<<ComboboxSelected>>", self.callbackFunc)
        combo.pack()
        self.combobox = combo

        applybtn = tkinter.Button(self.root, text="Apply",
                                 command=lambda: self.load(self.activeFilename, a, canvas, self.combobox.get()))
        applybtn.pack()

        if(self.activeCanvas is None):
            toolbar = NavigationToolbar2Tk(canvas, self.root)
            toolbar.update()
            canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)

    def callbackFunc(self, event):
        print("New Element is = "+ self.combobox.get())

    def browse(self,a,canvas,mode):
        filename = filedialog.askopenfilename(initialdir="/home/denizyu/Pictures", title="Select A File", filetypes=
        (("jpeg files", "*.jpg"), ("all files", "*.*")))
        self.activeFilename = filename
        self.load(filename,a,canvas,mode)

    def rgb2gray(rgb):
        return np.dot(rgb[..., :3], [0.299, 0.587, 0.144])

    def plot_img_and_hist(image, axes, bins=256):
        """Plot an image along with its histogram and cumulative histogram."""
        image = img_as_float(image)
        ax_img, ax_hist = axes
        ax_cdf = ax_hist.twinx()

        # Display image
        ax_img.imshow(image, cmap=plt.cm.gray)
        ax_img.set_axis_off()

        # Display histogram
        ax_hist.hist(image.ravel(), bins=bins, histtype='step', color='black')
        ax_hist.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
        ax_hist.set_xlabel('Pixel intensity')
        ax_hist.set_xlim(0, 1)
        ax_hist.set_yticks([])

        # Display cumulative distribution
        img_cdf, bins = exposure.cumulative_distribution(image, bins)
        ax_cdf.plot(bins, img_cdf, 'r')
        ax_cdf.set_yticks([])

        return ax_img, ax_hist, ax_cdf

    def load(self, filename,a,canvas,mode):
        imarray = mpimg.imread(filename)
        if mode == 'default':
            plt.imshow(imarray)
            t = np.arange(0, 3, .01)
            a.imshow(imarray)

        elif mode == 'gray':
            gray = rgb2gray(imarray)
            plt.imshow(imarray)
            t = np.arange(0, 3, .01)
            a.imshow(gray, cmap=plt.cm.gray)

        elif mode == 'very_blurry':
            very_blurred = ndimage.uniform_filter(imarray, size=(11, 11, 1))
            plt.imshow(imarray)
            t = np.arange(0, 3, .01)
            a.imshow(very_blurred)

        elif mode == 'sharp':
            sharp = unsharp_mask(imarray, radius=5, amount=2)
            plt.imshow(imarray)
            t = np.arange(0, 3, .01)
            a.imshow(sharp)

        elif mode == 'motion_blur':
            size = 50
            # generating the kernel
            kernel_motion_blur = np.zeros((size, size))
            kernel_motion_blur[int((size - 1) / 2), :] = np.ones(size)
            kernel_motion_blur = kernel_motion_blur / size

            # applying the kernel to the input image
            motionblur = cv2.filter2D(imarray, -1, kernel_motion_blur)
            plt.imshow(imarray)
            t = np.arange(0, 3, .01)
            a.imshow(motionblur)

        elif mode == 'find_edges':
            """does not work with 3d"""
            gray = rgb2gray(imarray)
            edgey = generic_gradient_magnitude(gray, sobel)
            plt.imshow(imarray)
            t = np.arange(0, 3, .01)
            a.imshow(edgey)

        elif mode == 'contrast':
            """does not work with 3d
            gray = rgb2gray(imarray)
            plt.imshow(gray, vmin=30, vmax=200)
            t = np.arange(0, 3, .01)
            a.imshow(gray, cmap="gray")"""
            # -----Converting image to LAB Color model-----------------------------------

        elif mode == 'mean':
            image = cv2.cvtColor(imarray, cv2.COLOR_BGR2HSV)  # convert to HSV
            figure_size = 9  # the dimension of the x and y axis of the kernal.
            new_image = cv2.blur(image, (figure_size, figure_size))
            plt.figure(figsize=(11, 6))
            plt.subplot(121), plt.imshow(cv2.cvtColor(image, cv2.COLOR_HSV2RGB)), plt.title('Original')
            plt.xticks([]), plt.yticks([])
            plt.subplot(122), plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_HSV2RGB)), plt.title('Mean filter')
            plt.xticks([]), plt.yticks([])
            plt.imshow(imarray)
            a.imshow(new_image)

        elif mode == 'contour':
            gray = rgb2gray(imarray)
            plt.imshow(imarray)
            plt.contour(gray, [50, 200])
            t = np.arange(0, 3, .01)
            a.imshow(gray, cmap="gray")

        elif mode == "lighten":
            image = cv2.add(imarray, np.array([50.0]))
            plt.imshow(image)
            a.imshow(image)

        elif mode == "darken":
            image = cv2.subtract(imarray, np.array([50.0]))
            plt.imshow(image)
            a.imshow(image)

        elif mode == "invert":
            image = np.invert(imarray)
            plt.imshow(image)
            a.imshow(image)

        elif mode == "gaussian":
            image = gaussian_filter(imarray, sigma=1)
            plt.imshow(image)
            a.imshow(image)

        elif mode == "percentile_mean":
            gray = rgb2gray(imarray)
            selem = disk(20)
            percentile_result = rank.mean_percentile(gray, selem=selem, p0=.1, p1=.9)
            plt.imshow(percentile_result)
            a.imshow(percentile_result, cmap="gray")

        elif mode == "bilateral_mean":
            gray = rgb2gray(imarray)
            selem = disk(20)
            bilateral_result = rank.mean_bilateral(gray, selem=selem, s0=500, s1=500)
            plt.imshow(bilateral_result)
            a.imshow(bilateral_result, cmap="gray")

        #uzaysal
        elif mode == 'crop':
            """does not work with 3d"""
            gray = rgb2gray(imarray)
            crop = gray[30:141, 15:140]
            plt.imshow(imarray)
            t = np.arange(0, 3, .01)
            a.imshow(crop, cmap="gray")

        elif mode == 'rescale':
            gray= rgb2gray(imarray)
            image_rescaled = rescale(gray, 0.25, anti_aliasing=False)
            plt.imshow(imarray)
            t = np.arange(0, 3, .01)
            a.imshow(image_rescaled, cmap='gray')

        elif mode == 'resize':
            gray= rgb2gray(imarray)
            image_resized = resize(gray, (gray.shape[0] // 4, gray.shape[1] // 4),
                                   anti_aliasing=True)
            plt.imshow(imarray)
            t = np.arange(0, 3, .01)
            a.imshow(image_resized, cmap='gray')

        elif mode == 'downscaled':
            gray= rgb2gray(imarray)
            image_downscaled = downscale_local_mean(gray, (4, 3))
            plt.imshow(imarray)
            t = np.arange(0, 3, .01)
            a.imshow(image_downscaled, cmap='gray')

        elif mode == 'rotate':
            new_pic = rotate(imarray, 180)
            plt.imshow(imarray)
            t = np.arange(0, 3, .01)
            a.imshow(new_pic)

        elif mode == 'swirl':
            swirled = swirl(imarray, rotation=0, strength=10, radius=250)
            plt.imshow(imarray)
            t = np.arange(0, 3, .01)
            a.imshow(swirled)

        #intensity adjustment
        elif mode == 'intensity':
            new_image = cv2.convertScaleAbs(imarray, alpha=1, beta=2.54)
            plt.imshow(imarray)
            t = np.arange(0, 3, .01)
            a.imshow(new_image)

        elif mode == 'histogram':
            """im = cv2.imread(filename)
            vals=im.mean(axis=2).flatten()
            counts, bins = np.histogram(vals,range(257))
            plt.bar(bins[:-1] - 0.5, counts, width=1 , edgecolor="none")
            plt.xlim([-0.5, 255.5])
            plt.show()"""
            img = plt.imread(filename)  # reads image data
            plt.hist(img.flatten(), 256, [0, 256]);
            plt.show()



        canvas.draw()

    def _quit(self):
        self.root.quit()     # stops mainloop
        self.root.destroy()  # this is necessary on Windows to prevent

GUI()