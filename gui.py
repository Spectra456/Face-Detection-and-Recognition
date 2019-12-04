import tkinter as tk
from tkinter import Label, Canvas, YES, BOTH, Frame, N, W, Scrollbar, ttk
from PIL import Image, ImageTk
from video_capture import *


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

def add_face(frame_log, counter, frame ,face_locations, size=(128, 128)):
	face_cropped = Image.fromarray(frame[face_locations[0]:face_locations[2], face_locations[3]:face_locations[1]]).resize(size)
	face_cropped_tk = ImageTk.PhotoImage(image=face_cropped)
	log_frame=Label(frame_log, image=face_cropped_tk)
	log_frame.imgtk = face_cropped_tk
	log_frame.configure(image=face_cropped_tk)
	log_frame.grid(row=counter, column=0)
	return log_frame

def add_meta(frame_log, face_name,face_features,face_emotion,counter, witdh_size = 30):
	label = Label(frame_log, width=witdh_size, text='Name: {} \nGender: {} \nAge: {}\nRace: {}\nGlass:{}\n Emotion:{} '.format(face_name,face_features[2],face_features[0],face_features[1],face_features[3],face_emotion)).grid(row=counter, column=1)
	return label

def add_frame(frame_window,frame, size=(1080, 720)):
	cv2image = change_color_dimension(frame)
	img = Image.fromarray(cv2image).resize(size)
	img_tk = ImageTk.PhotoImage(image=img)
	frame_window.configure(image=img_tk)
	frame_window.imgtk = img_tk

	return frame_window
