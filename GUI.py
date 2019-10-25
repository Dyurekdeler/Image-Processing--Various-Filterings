from tkinter import *
from tkinter import filedialog
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure

class MainMenu():
    def __init__(self):
        self.window = Tk()
        self.window.title("Welcome to LikeGeeks app")
        self.window.geometry('350x200')

        self.closebutton = Button(self.window, text="Exit", command=self.close)
        self.closebutton.pack(side='bottom')
        browsebutton = Button(self.window, text="Browse", command=self.browse)
        self.browsebutton.pack()
        self.window.mainloop()
    def close(self):
        self.window.destroy()
    def browse(self):
        self.filename = filedialog.askopenfilename(initialdir="/", title="Select A File", filetype=
        (("jpeg files", "*.jpg"), ("all files", "*.*")))
    def matplotCanvas(self):
        f = Figure(figsize=())

MainMenu()