from tkinter import Frame
from tkinter import Tk
from tkinter import filedialog
import tkinter as tki
from tkinter import messagebox


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from numpy import array, arange, sin, pi

class Interface():

    def __init__(self):
        #initialize a window and its components
        self.root = Tk()
        self.currImg = None
        root_panel = Frame(self.root)
        root_panel.pack(side="bottom", fill="both", expand="yes")

        btn_panel = Frame(root_panel, height=35)
        btn_panel.pack(side='top', fill="both", expand="yes")
        browsebtn = tki.Button(self.root, text="Browse A File", command=self.browse)
        browsebtn.pack()
        exitbtn = tki.Button(self.root, text="Exit",command=self.root.destroy)
        exitbtn.pack()
        savebtn = tki.Button(self.root, text="Save Image", command=self.save)
        savebtn.pack()

        self.root.minsize(400, 200)
        self.root.mainloop()

    def browse(self):
        filename = filedialog.askopenfilename(initialdir="/", title="Select A File", filetype=
        (("jpeg files", "*.jpg"), ("all files", "*.*")))
        self.load(filename)

    def load(self, filename):
        #loads an image browsed by user
        img_arr = mpimg.imread(filename)
        f = Figure()
        a = f.add_subplot(111)
        plt.imshow(img_arr)
        canvas = FigureCanvasTkAgg(f, master=self.root)
        canvas.draw()
        canvas.get_tk_widget().pack(side="top", fill="both", expand=1)
        canvas._tkcanvas.pack(side="top", fill="both", expand=1)
        self.currImg = filename

    def save(self):
        if self.currImg is not None:
            newfilename = self.currImg + 'newly.png'
            alertmsg = "Image has been saved to " + newfilename
            plt.savefig(newfilename)
            messagebox.showinfo("Information", alertmsg)
Interface()