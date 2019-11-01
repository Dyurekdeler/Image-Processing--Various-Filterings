import tkinter
from tkinter import filedialog

import cv2
from scipy import ndimage
from scipy.ndimage import generic_gradient_magnitude
from skimage.color import rgb2gray
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import numpy as np
from skimage.filters import unsharp_mask, sobel


class GUI():

    def __init__(self):
        self.root = tkinter.Tk()
        self.root.wm_title("Densy's Image World")
        self.activeCanvas = None
        self.activeFilename = None
        self.createWidgets()
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

        graybtn = tkinter.Button(self.root, text="Convert Gray", command= lambda: self.load(self.activeFilename, a, canvas, 'gray'))
        graybtn.pack()

        veryblurrybtn = tkinter.Button(self.root, text="Blur A Lot",
                                 command=lambda: self.load(self.activeFilename, a, canvas, 'very_blurry'))
        veryblurrybtn.pack()

        sharpbtn = tkinter.Button(self.root, text="Sharpen",
                                 command=lambda: self.load(self.activeFilename, a, canvas, 'sharp'))
        sharpbtn.pack()

        motionblurbtn = tkinter.Button(self.root, text="Motion Blur",
                                 command=lambda: self.load(self.activeFilename, a, canvas, 'motion_blur'))
        motionblurbtn.pack()

        contrastbtn = tkinter.Button(self.root, text="Contrast",
                                       command=lambda: self.load(self.activeFilename, a, canvas, 'contrast'))
        contrastbtn.pack()

        contourbtn = tkinter.Button(self.root, text="Contour Lines",
                                       command=lambda: self.load(self.activeFilename, a, canvas, 'contour'))
        contourbtn.pack()
        edgebtn = tkinter.Button(self.root, text="Find Edges",
                                       command=lambda: self.load(self.activeFilename, a, canvas, 'find_edges'))
        edgebtn.pack()
        cropbtn = tkinter.Button(self.root, text="Crop",
                                 command=lambda: self.load(self.activeFilename, a, canvas, 'crop'))
        cropbtn.pack()
        if(self.activeCanvas is None):
            toolbar = NavigationToolbar2Tk(canvas, self.root)
            toolbar.update()
            canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)

    def browse(self,a,canvas,mode):
        filename = filedialog.askopenfilename(initialdir="/", title="Select A File", filetype=
        (("jpeg files", "*.jpg"), ("all files", "*.*")))
        self.activeFilename = filename
        self.load(filename,a,canvas,mode)

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
            edgey = generic_gradient_magnitude(imarray, sobel)
            plt.imshow(imarray)
            t = np.arange(0, 3, .01)
            a.imshow(edgey)

        elif mode == 'contrast':
            plt.imshow(imarray, vmin=30, vmax=200)
            t = np.arange(0, 3, .01)
            a.imshow(imarray)

        elif mode == 'contour':
            plt.imshow(imarray)
            plt.contour(imarray, [50, 200])
            t = np.arange(0, 3, .01)
            a.imshow(imarray)
        #HISTOGRAM
        elif mode == 'crop':
            crop = imarray[77:141, 57:121]
            plt.imshow(imarray)
            plt.contour(imarray, [50, 200])
            t = np.arange(0, 3, .01)
            a.imshow(crop)
        canvas.draw()



    def _quit(self):
        self.root.quit()     # stops mainloop
        self.root.destroy()  # this is necessary on Windows to prevent
                        # Fatal Python Error: PyEval_RestoreThread: NULL tstate

        # If you put root.destroy() here, it will cause an error if the window is
        # closed with the window manager.

GUI()