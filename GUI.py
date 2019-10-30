from tkinter import Frame, Tk

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from numpy import array, arange, sin, pi

root = Tk()
root_panel = Frame(root)
root_panel.pack(side="bottom", fill="both", expand="yes")

btn_panel = Frame(root_panel, height=35)
btn_panel.pack(side='top', fill="both", expand="yes")

img_arr = mpimg.imread('winnie.png')
imgplot = plt.imshow(img_arr)

#here is the example of how I embed matplotlib graph in Tkinter,
#basically, I want to do the same with the image (imgplot)
f = Figure()
a = f.add_subplot(111)
t = arange(0.0, 3.0, 0.01)
s = sin(2*pi*t)
a.imshow(img_arr)

canvas = FigureCanvasTkAgg(f, master=root)
canvas.draw()
canvas.get_tk_widget().pack(side="top", fill="both", expand=1)
canvas._tkcanvas.pack(side="top", fill="both", expand=1)

root.mainloop()