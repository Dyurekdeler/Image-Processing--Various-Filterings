import tkinter
from tkinter import filedialog

from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import numpy as np

class GUI():

    def __init__(self):
        self.root = tkinter.Tk()
        self.root.wm_title("Densy's Image World")

        self.canvas = FigureCanvasTkAgg(self.load(), master=self.root)  # A tk.DrawingArea.
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)

        self.toolbar = NavigationToolbar2Tk(self.canvas, self.root)
        self.toolbar.update()
        self.canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)

        self.canvas.mpl_connect("key_press_event", self.on_key_press)

        quitbtn = tkinter.Button(master=self.root, text="Quit", command=self._quit)
        quitbtn.pack(side=tkinter.BOTTOM)

        browsebtn = tkinter.Button(self.root, text="Browse A File", command=self.browse)
        browsebtn.pack()

        tkinter.mainloop()

    def load(self, filename):
        imarray = mpimg.imread('winnie.png')
        plt.imshow(imarray)
        fig = Figure(figsize=(5, 4), dpi=100)
        t = np.arange(0, 3, .01)
        # fig.add_subplot(111).plot(t, 2 * np.sin(2 * np.pi * t))
        a = fig.add_subplot(111)
        a.imshow(imarray)
        return fig

    def browse(self):
        filename = filedialog.askopenfilename(initialdir="/", title="Select A File", filetype=
        (("jpeg files", "*.jpg"), ("all files", "*.*")))
        self.load(filename)

    def on_key_press(self, event):
        print("you pressed {}".format(event.key))
        key_press_handler(event, self.canvas, self.toolbar)

    def _quit(self):
        self.root.quit()     # stops mainloop
        self.root.destroy()  # this is necessary on Windows to prevent
                        # Fatal Python Error: PyEval_RestoreThread: NULL tstate

        # If you put root.destroy() here, it will cause an error if the window is
        # closed with the window manager.

GUI()