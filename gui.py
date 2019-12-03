import tkinter as tk
from tkinter import Label, Canvas, YES, BOTH, Frame, N, W, Scrollbar, ttk


def get_mainWindow(max_width = 1500, max_height = 720):
	mainWindow = tk.Tk()
	mainWindow.configure()
	mainWindow.geometry('%dx%d+%d+%d' % (max_width,max_height,0,0))
	mainWindow.resizable(0,0)

	return mainWindow

def get_logScrollbar():
	canvas = Canvas(width=300, height=140, bg='gray',scrollregion=(0,0,100,20000))
	canvas.pack(expand=YES, fill=BOTH)
	frame = Frame(canvas)
	canvas.create_window(1081, 0, window=frame, width=0, height=999999999999999, anchor=N+W)
	canvas.sbar = Scrollbar(orient='vertical')
	canvas['yscrollcommand'] = canvas.sbar.set
	canvas.sbar['command'] = canvas.yview
	canvas.sbar.place(x=1490,y=0, height=720)

	return canvas, frame

def get_frameWindow(main_Window):
	main_Frame = ttk.Frame(main_Window)
	main_Frame.place(x=0, y=0)
	lmain = Label(main_Frame)
	lmain.grid(row=0, column=0)

	return lmain