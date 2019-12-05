from PIL import Image, ImageTk
import os
import warnings
from random import randint
from face_detection import Face_Detection
from drawer import *
from gui import *
from video_capture import *
from drawer import *
from face_recognitor import *
from face_features import *
from face_emotions import *
import copy
import time

warnings.filterwarnings('ignore')

counter = 0
counterFrame = 24
counterSkip = 0
#Graphics window

main_window = get_mainWindow()
canvas, frame = get_logScrollbar()
frame_window = get_frameWindow(main_window)

# for face detection
frame_handler = Frame_Handler(0)

yolo = Face_Detection(0.5,0.5, False)
face_recognition = Face_Recognition()
face_features = Face_Features()
face_emotions = Emotion_Recognition()

def show_frame():
	start = time.time()

	global counter

	global counterFrame
	counterFrame += 1

	global counterSkip
	counterSkip +=1

	flag_scroll = False

	frame_capture = frame_handler.get_frame()
	faceViewImage = change_color_dimension(frame_capture)
	frame_gray = change_color_dimension_to_gray(frame_capture)

	# detect faces using dlib detector
	detected = yolo.detect(frame_capture)

	face_encodings = face_recognition.get_encodings(frame_capture, detected)
	face_features_data = face_features.predict(face_encodings)

	face_locations = []
	face_names = []
	face_features_array = []
	face_emotions_array = []

	if counterFrame % 48 == 0:
		flag_scroll = True

		counterFrame = 0
	else:
		flag_scroll = False

	if counterSkip  == 48:
		canvas.yview_scroll(3, "units")
		counterSkip = 0

	for i, d in enumerate(detected):
		x1, y1, x2, y2 = d[3], d[0], d[1], d[2]
		cv2.rectangle(frame_capture, (x1, y1), (x2, y2), (0, 255, 0), 2)
		face_locations.append([y1, x2, y2, x1])
		if flag_scroll == True:
			face_names.append(face_recognition.predict(face_encodings[i]))
			face_features_array.append(face_features.check_prediction(face_features_data.loc[i]))
			face_emotions_array.append(face_emotions.predict(frame_gray, detected[i]))

	# draw results
	if flag_scroll == True:
		for i, d in enumerate(detected):

			log_frame = add_face(frame, counter, faceViewImage, face_locations[i])
			add_meta(frame, face_names[i], face_features_array[i], face_emotions_array[i], counter)
			counter+=1
			
	frame_window1 = add_frame(frame_window, frame_capture)

	print('Done. (%.3fs)' % (time.time() - start), flush=True)

	frame_window1.after(1, show_frame)

show_frame()  #Display
main_window.mainloop()  #Starts GUI
