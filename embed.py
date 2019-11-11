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

        intensitylabel = tkinter.Label(self.root, text="Enter intensity level")
        intensitylabel.pack()

        self.intelvl = tkinter.StringVar()
        tkinter.Entry(self.root, textvariable=self.intelvl).pack()

        combo = ttk.Combobox(self.root,
                                    values=[
                                        "gray","very_blurry","sharp","mean",
                                        "motion_blur","lighten","darken","invert","gaussian",
                                        "percentile_mean","bilateral_mean",'find_edges',

                                        "crop","rescale","resize","downscaled","rotate","swirl",
                                    
                                        "intensity",

                                        "histogram","equalization",

                                        "erosion","dilation","opening","closing","morp_gradient","top_hat","black_hat"])
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
        print("New Element is = "+ self.combobox.get()+" : "+ self.intelvl.get())

    def browse(self,a,canvas,mode):
        filename = filedialog.askopenfilename(initialdir="/home/denizyu/Pictures", title="Select A File", filetypes=
        (("jpeg files", "*.jpg"), ("all files", "*.*")))
        self.activeFilename = filename
        if filename.endswith('.mp4'):
            self.loadvideo(filename,a,canvas)
        else:
            self.load(filename,a,canvas,mode)

    def load(self, filename,a,canvas,mode):
        imarray = mpimg.imread(filename)
        plt.close('all')
        if mode == 'default':
            a.imshow(imarray)

        #IMG FILTERING
        elif mode == 'gray':
            gray = rgb2gray(imarray)
            a.imshow(gray, cmap=plt.cm.gray)

        elif mode == 'very_blurry':
            very_blurred = ndimage.uniform_filter(imarray, size=(11, 11, 1))
            a.imshow(very_blurred)

        elif mode == 'sharp':
            sharp = unsharp_mask(imarray, radius=5, amount=2)
            a.imshow(sharp)

        elif mode == 'motion_blur':
            size = 50
            # generating the kernel
            kernel_motion_blur = np.zeros((size, size))
            kernel_motion_blur[int((size - 1) / 2), :] = np.ones(size)
            kernel_motion_blur = kernel_motion_blur / size

            # applying the kernel to the input image
            motionblur = cv2.filter2D(imarray, -1, kernel_motion_blur)
            a.imshow(motionblur)

        elif mode == 'mean':
            image = cv2.cvtColor(imarray, cv2.COLOR_BGR2HSV)  # convert to HSV
            figure_size = 9  # the dimension of the x and y axis of the kernal.
            new_image = cv2.blur(image, (figure_size, figure_size))
            plt.figure(figsize=(11, 6))
            plt.subplot(121), plt.imshow(cv2.cvtColor(image, cv2.COLOR_HSV2RGB)), plt.title('Original')
            plt.xticks([]), plt.yticks([])
            plt.subplot(122), plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_HSV2RGB)), plt.title('Mean filter')
            plt.xticks([]), plt.yticks([])
            a.imshow(new_image)

        elif mode == 'find_edges':
            gray = rgb2gray(imarray)
            edge = sobel(gray)
            a.imshow(edge)

        elif mode == "lighten":
            image = cv2.add(imarray, np.array([50.0]))
            a.imshow(image)

        elif mode == "darken":
            image = cv2.subtract(imarray, np.array([50.0]))
            a.imshow(image)

        elif mode == "invert":
            image = np.invert(imarray)
            a.imshow(image)

        elif mode == "gaussian":
            image = gaussian_filter(imarray, sigma=1)
            a.imshow(image)

        elif mode == "percentile_mean":
            gray = rgb2gray(imarray)
            selem = disk(20)
            percentile_result = rank.mean_percentile(gray, selem=selem, p0=.1, p1=.9)
            a.imshow(percentile_result, cmap="gray")

        elif mode == "bilateral_mean":
            gray = rgb2gray(imarray)
            selem = disk(20)
            bilateral_result = rank.mean_bilateral(gray, selem=selem, s0=500, s1=500)
            a.imshow(bilateral_result, cmap="gray")

        #SPATIAL
        elif mode == 'crop':
            """does not work with 3d"""
            gray = rgb2gray(imarray)
            crop = gray[30:141, 15:140]
            a.imshow(crop, cmap="gray")

        elif mode == 'rescale':
            gray= rgb2gray(imarray)
            image_rescaled = rescale(gray, 0.25, anti_aliasing=False)
            a.imshow(image_rescaled, cmap='gray')

        elif mode == 'resize':
            gray= rgb2gray(imarray)
            image_resized = resize(gray, (gray.shape[0] // 4, gray.shape[1] // 4),
                                   anti_aliasing=True)
            a.imshow(image_resized, cmap='gray')

        elif mode == 'downscaled':
            gray= rgb2gray(imarray)
            image_downscaled = downscale_local_mean(gray, (4, 3))
            a.imshow(image_downscaled, cmap='gray')

        elif mode == 'rotate':
            new_pic = rotate(imarray, 180)
            a.imshow(new_pic)

        elif mode == 'swirl':
            swirled = swirl(imarray, rotation=0, strength=10, radius=250)
            a.imshow(swirled)

        #INTENSITY ADJUSTMENT
        elif mode == 'intensity':
            gamma = float(self.intelvl.get())
            gamma_corrected = np.array(255*(imarray / 255) ** gamma, dtype = 'uint8')
            a.imshow(gamma_corrected)

        #HISTOGRAM AND EQUALIZATION
        elif mode == 'histogram':
            img = cv2.imread(filename, 0)
            gray = rgb2gray(img)
            plt.hist(gray.ravel(), 256, [0, 256])
            plt.show()

        elif mode == 'equalization':
            img = cv2.imread(filename, 0)
            gray = rgb2gray(img)
            equ = cv2.equalizeHist(gray)
            a.imshow(equ, cmap='gray')
            plt.hist(equ.ravel(), 256, [0, 256])
            plt.show()

        #MORPHOLOGICAL
        elif mode == 'erosion':
            kernel = np.ones((5, 5), np.uint8)
            erosion = cv2.erode(imarray, kernel, iterations=1)
            a.imshow(erosion)

        elif mode == 'dilation':
            kernel = np.ones((5, 5), np.uint8)
            dilation = cv2.dilate(imarray,kernel,iterations = 1)
            a.imshow(dilation)

        elif mode == 'opening':
            kernel = np.ones((5, 5), np.uint8)
            erosion = cv2.erode(imarray, kernel, iterations=1)
            a.imshow(erosion)

        elif mode == 'closing':
            kernel = np.ones((5, 5), np.uint8)
            closing = cv2.morphologyEx(imarray, cv2.MORPH_CLOSE, kernel)
            a.imshow(closing)

        elif mode == 'morp_gradient':
            kernel = np.ones((5, 5), np.uint8)
            gradient = cv2.morphologyEx(imarray, cv2.MORPH_GRADIENT, kernel)
            a.imshow(gradient)

        elif mode == 'top_hat':
            kernel = np.ones((5, 5), np.uint8)
            tophat = cv2.morphologyEx(imarray, cv2.MORPH_TOPHAT, kernel)
            a.imshow(tophat)

        elif mode == 'black_hat':
            kernel = np.ones((5, 5), np.uint8)
            blackhat = cv2.morphologyEx(imarray, cv2.MORPH_BLACKHAT, kernel)
            a.imshow(blackhat)

        canvas.draw()

    def loadvideo(self,filename,a,canvas):
        cap = cv2.VideoCapture(filename)
        # Read until video is completed
        while (cap.isOpened()):
            # Capture frame-by-frame
            ret, frame = cap.read()
            if ret == True:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                blurred = cv2.GaussianBlur(frame, (5,5), 0)
                laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
                canny = cv2.Canny(blurred,100,150)

                # Display the resulting frame
                numpy_horizontal = np.hstack((laplacian, canny))
                numpy_horizontal_concat = np.concatenate((laplacian, canny), axis=1)
                cv2.imshow('Original and Laplacian and Canny', numpy_horizontal)
                cv2.imshow('Original', frame)

                # Press Q on keyboard to  exit
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break

            # Break the loop
            else:
                break

        # When everything done, release the video capture object
        cap.release()
    def _quit(self):
        self.root.quit()     # stops mainloop
        self.root.destroy()  # this is necessary on Windows to prevent

GUI()