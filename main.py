from PIL import Image, ImageTk
from random import randint
import warnings
import copy
import time
import os

from face_detection import Face_Detection
from face_recognitor import *
from video_capture import *
from face_features import *
from face_emotions import *
from drawer import *
from gui import *

warnings.filterwarnings('ignore')

# Counters for gui
counter = 0
counterFrame = 24
counterSkip = 0

# Drawing gui
main_window = get_mainWindow()
canvas, frame = get_logScrollbar()
frame_window = get_frameWindow(main_window)

# Init frame handler for capture frames from sources
frame_handler = Frame_Handler('/home/spectra/Downloads/test_walk.MOV')

# Init face detector
yolo = Face_Detection(0.7,0.7, False)
# Init face recognition
face_recognition = Face_Recognition()
# Init face features recognition
face_features = Face_Features()
# Init emotion recognition
face_emotions = Emotion_Recognition()

# Function for drawing gui per frame
def show_frame():	

	# For counting fps
	start = time.time()

	global counter

	global counterFrame
	counterFrame += 1

	global counterSkip
	counterSkip +=1

	flag_scroll = False

	# Get frame from source 
	frame_capture = frame_handler.get_frame()
	# BGR to RGB
	faceViewImage = change_color_dimension(frame_capture)
	# BGR to GRY
	frame_gray = change_color_dimension_to_gray(frame_capture)

	# detect faces using yolo detector
	detected = yolo.detect(frame_capture)
	# get encodings of detected faces
	face_encodings = face_recognition.get_encodings(frame_capture, detected)
	# get features of faces from encodings
	face_features_data = face_features.predict(face_encodings)

	face_locations = []
	face_names = []
	face_features_array = []
	face_emotions_array = []

	# For scrolling scrollbar and adding faces to scrollbar
	if counterFrame % 48 == 0:
		flag_scroll = True

		counterFrame = 0
	else:
		flag_scroll = False

	if counterSkip  == 48:
		canvas.yview_scroll(3, "units")
		counterSkip = 0

	for i, d in enumerate(detected):
		# Draw rectangles of detected on frame
		x1, y1, x2, y2 = d[3], d[0], d[1], d[2]
		cv2.rectangle(frame_capture, (x1, y1), (x2, y2), (0, 255, 0), 2)
		face_locations.append([y1, x2, y2, x1])

		if flag_scroll == True:
			# Find names in DB based on encodings
			face_names.append(face_recognition.predict(face_encodings[i]))
			# Check features predicition
			face_features_array.append(face_features.check_prediction(face_features_data.loc[i]))
			# Predict emotion of face
			face_emotions_array.append(face_emotions.predict(frame_gray, detected[i]))

	# Add faces and meta information to scrollbar(log)
	if flag_scroll == True:
		for i, d in enumerate(detected):

			log_frame = add_face(frame, counter, faceViewImage, face_locations[i])
			add_meta(frame, face_names[i], face_features_array[i], face_emotions_array[i], counter)
			counter+=1
			
	frame_window1 = add_frame(frame_window, frame_capture)

	print('Done. (%.3fs)' % (time.time() - start))

	frame_window1.after(1, show_frame)

show_frame()  #Display
main_window.mainloop()  #Starts GUI
