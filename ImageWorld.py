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
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
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
        tkinter.mainloop()

    def createWidgets(self):

        fig = Figure(figsize=(5, 4), dpi=100)
        a = fig.add_subplot(111)
        canvas = FigureCanvasTkAgg(fig, master=self.root)  # A tk.DrawingArea.
        canvas.draw()
        canvas.get_tk_widget().pack(side=tkinter.RIGHT, fill=tkinter.BOTH, expand=1)

        browseframe = tkinter.Frame(self.root)
        browseframe.pack()

        browselabel = tkinter.Label(browseframe, text="Please Choose a File")
        browselabel.pack(side=tkinter.TOP)

        browsebtn = tkinter.Button(browseframe, text="Browse an Image", command= lambda: self.browse(a, canvas, 'default','img'))
        browsebtn.pack(side=tkinter.LEFT)

        browsevidbtn = tkinter.Button(browseframe, text="Browse an Video", command=lambda: self.browse(a, canvas, 'default','vid'))
        browsevidbtn.pack(side=tkinter.RIGHT)

        basicsframe = tkinter.Frame(self.root)
        basicsframe.pack()

        basicslabel = tkinter.Label(basicsframe, text="Choose an Filter")
        basicslabel.pack(side=tkinter.TOP)

        combo = ttk.Combobox(basicsframe,
                                    values=[
                                        "gray","blurry","sharp","mean",
                                        "motion_blur","lighten","darken","invert","gaussian",
                                        "percentile_mean","bilateral_mean",'find_edges',

                                        "intensity",

                                        "rescale","resize","downscaled","rotate","swirl",

                                        "histogram","equalization",

                                        "erosion","dilation","opening","closing","morp_gradient","top_hat","black_hat"])
        combo.current(0)
        combo.bind("<<ComboboxSelected>>", self.callbackFunc)
        combo.pack(side=tkinter.LEFT)
        self.combobox = combo

        applybtn = tkinter.Button(basicsframe, text="Apply",
                                 command=lambda: self.load(self.activeFilename, a, canvas, self.combobox.get()))
        applybtn.pack(side=tkinter.RIGHT)

        self.optional = tkinter.StringVar()
        entry = tkinter.Entry(basicsframe, textvariable=self.optional)
        entry.insert(0, 'Optinal Value')
        entry.pack(side=tkinter.LEFT)

        cropxframe = tkinter.Frame(self.root)
        cropxframe.pack()
        cropyframe = tkinter.Frame(self.root)
        cropyframe.pack()
        xlabel = tkinter.Label(cropxframe, text="Enter X Interval")
        xlabel.pack(side=tkinter.TOP)

        ylabel = tkinter.Label(cropyframe, text="Enter Y Interval")
        ylabel.pack(side=tkinter.TOP)

        self.x1 = tkinter.StringVar()
        tkinter.Entry(cropxframe, textvariable=self.x1).pack(side=tkinter.LEFT)
        self.x2 = tkinter.StringVar()
        tkinter.Entry(cropxframe, textvariable=self.x2).pack(side=tkinter.RIGHT)
        self.y1 = tkinter.StringVar()
        tkinter.Entry(cropyframe, textvariable=self.y1).pack(side=tkinter.LEFT)
        self.y2 = tkinter.StringVar()
        tkinter.Entry(cropyframe, textvariable=self.y2).pack(side=tkinter.RIGHT)

        cropbtn = tkinter.Button(self.root, text="Crop Image", command=lambda: self.load(self.activeFilename,a, canvas, 'crop'))
        cropbtn.pack()

        originalbtn = tkinter.Button(self.root, text="Set Back Original Image", command= lambda : self.load(self.activeFilename, a, canvas, 'default'))
        originalbtn.pack()

        quitbtn = tkinter.Button(master=self.root, text="Quit", command=self._quit)
        quitbtn.pack(side=tkinter.BOTTOM)

        if(self.activeCanvas is None):
            toolbar = NavigationToolbar2Tk(canvas, self.root)
            toolbar.update()
            canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)

    def callbackFunc(self, event):
        print("New Element is = "+ self.combobox.get()+" optional value is : "+ self.optional.get())

    def browse(self,a,canvas,mode, filetype):
        if filetype == 'img':
            filename = filedialog.askopenfilename(initialdir="/home/denizyu/Pictures", title="Select A File", filetypes=
            (("jpeg files", "*.jpg"), ("png files", "*.png")))
            self.load(filename, a, canvas, mode)
        else:
            filename = filedialog.askopenfilename(initialdir="/home/denizyu/Pictures", title="Select A File", filetypes=(("MP4 files", "*.mp4"),
                                                                                         ("WMV files", "*.wmv"), ("AVI files", "*.avi")))
            self.loadvideo(filename, a, canvas)
        self.activeFilename = filename

    def load(self, filename,a,canvas,mode):
        imarray = mpimg.imread(filename)
        plt.close('all')

        if mode == 'default':
            a.imshow(imarray)


        #IMG FILTERING
        elif mode == 'gray':
            gray = rgb2gray(imarray)
            a.imshow(gray, cmap=plt.cm.gray)

        elif mode == 'blurry':
            value=11
            if self.RepresentsFloat(self.optional.get()):
                value = int(self.optional.get())

            blurred = ndimage.uniform_filter(imarray, size=(value, value, 1))
            a.imshow(blurred)

        elif mode == 'sharp':
            # optional
            sharp = unsharp_mask(imarray, radius=5, amount=2)
            a.imshow(sharp)

        elif mode == 'motion_blur':
            size = 50
            if self.RepresentsFloat(self.optional.get()):
                size = int(self.optional.get())

            # generating the kernel
            kernel_motion_blur = np.zeros((size, size))
            kernel_motion_blur[int((size - 1) / 2), :] = np.ones(size)
            kernel_motion_blur = kernel_motion_blur / size

            # applying the kernel to the input image
            motionblur = cv2.filter2D(imarray, -1, kernel_motion_blur)
            a.imshow(motionblur)

        elif mode == 'mean':
            size = 9
            if self.RepresentsFloat(self.optional.get()):
                size = int(self.optional.get())
            new_image = cv2.medianBlur(imarray, size)
            a.imshow(new_image)

        elif mode == 'find_edges':
            gray = rgb2gray(imarray)
            edge = sobel(gray)
            a.imshow(edge)

        elif mode == "lighten":
            size = 50.0
            if self.RepresentsFloat(self.optional.get()):
                size = float(self.optional.get())
            image = cv2.add(imarray, np.array([size]))
            a.imshow(image)

        elif mode == "darken":
            size = 50.0
            if self.RepresentsFloat(self.optional.get()):
                size = float(self.optional.get())
            image = cv2.subtract(imarray, np.array([size]))
            a.imshow(image)

        elif mode == "invert":
            image = np.invert(imarray)
            a.imshow(image)

        elif mode == "gaussian":
            size = 25
            if self.RepresentsFloat(self.optional.get()):
                size = int(self.optional.get())
            image = cv2.GaussianBlur(imarray, (size, size), 0)
            a.imshow(image)

        elif mode == "percentile_mean":
            ##### bu nedir böyle ######
            gray = rgb2gray(imarray)
            selem = disk(20)
            percentile_result = rank.mean_percentile(gray, selem=selem, p0=.1, p1=.9)
            a.imshow(percentile_result, cmap="gray")

        elif mode == "bilateral_mean":
            size = 30
            if self.RepresentsFloat(self.optional.get()):
                size = int(self.optional.get())
            new_image = cv2.bilateralFilter(imarray, size, size * 2, size / 2)
            a.imshow(new_image)

        #SPATIAL
        elif mode == 'crop':
            axes_x1 = 30
            axes_x2 = 140
            axes_y1 = 15
            axes_y2 = 140
            if self.RepresentsFloat(self.x1.get()) and self.RepresentsFloat(self.x2.get()) and self.RepresentsFloat(self.y1.get()) and self.RepresentsFloat(self.y2.get()) :
                axes_x1 = int(self.x1.get())
                axes_x2 = int(self.x2.get())
                axes_y1 = int(self.y1.get())
                axes_y2 = int(self.y2.get())
            gray = rgb2gray(imarray)
            crop = gray[axes_x1:axes_x2, axes_y1:axes_y2]
            a.imshow(crop, cmap="gray")

        elif mode == 'rescale':
            size = 0.25
            if self.RepresentsFloat(self.optional.get()):
                size = float(self.optional.get())
            gray= rgb2gray(imarray)
            image_rescaled = rescale(gray, size, anti_aliasing=False)
            a.imshow(image_rescaled, cmap='gray')

        elif mode == 'resize':
            size = 4
            if self.RepresentsFloat(self.optional.get()):
                size = int(self.optional.get())

            gray = rgb2gray(imarray)
            image_resized = resize(gray, (gray.shape[0] // size, gray.shape[1] // size),
                                   anti_aliasing=True)
            a.imshow(image_resized, cmap='gray')

        elif mode == 'downscaled':
            gray= rgb2gray(imarray)
            image_downscaled = downscale_local_mean(gray, (4, 3))
            a.imshow(image_downscaled, cmap='gray')

        elif mode == 'rotate':
            size = 180
            if self.RepresentsFloat(self.optional.get()):
                size = int(self.optional.get())
            new_pic = rotate(imarray, size)
            a.imshow(new_pic)

        elif mode == 'swirl':
            size = 250
            if self.RepresentsFloat(self.optional.get()):
                size = int(self.optional.get())
            swirled = swirl(imarray, rotation=0, strength=10, radius=size)
            a.imshow(swirled)

        #INTENSITY ADJUSTMENT
        elif mode == 'intensity':
            gamma = 1.5
            if self.RepresentsFloat(self.optional.get()):
                gamma = float(self.optional.get())
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
            opening = cv2.morphologyEx(imarray, cv2.MORPH_OPEN, kernel)
            a.imshow(opening)

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
                blurred = cv2.GaussianBlur(frame, (5, 5), 0)
                laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
                canny = cv2.Canny(blurred, 100, 150)

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

    def RepresentsFloat(self, s):
        try:
            float(s)
            return True
        except ValueError:
            return False


GUI()