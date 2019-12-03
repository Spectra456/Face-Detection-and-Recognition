from keras.models import load_model
from keras.models import model_from_json
import numpy as np
import cv2

class Emotion_Recognition:
	def __init__(self, model_path='models/fer2013_mini_XCEPTION.46-0.82.hdf5'):
		emotion_model_path = model_path
		self.emotion_labels = {0: 'angry', 1: 'happy', 2: 'surprise', 3: 'neutral'}
		self.emotion_classifier = load_model(emotion_model_path, compile=False)
		self.emotion_target_size = self.emotion_classifier.input_shape[1:3]
		self.emotion_offsets = (20, 40)

	def predict(self, gray, face_location):
		try:
			x1, x2, y1, y2 = apply_offsets(face_location, self.emotion_offsets)
			gray_face = gray[y1:y2, x1:x2]
			gray_face = cv2.resize(gray_face, (self.emotion_target_size))
			gray_face = preprocess_input(gray_face, False)
			gray_face = np.expand_dims(gray_face, 0)
			gray_face = np.expand_dims(gray_face, -1)
			emotion_label_arg = np.argmax(self.emotion_classifier.predict(gray_face))
			emotion_text = self.emotion_labels[emotion_label_arg]

			return emotion_text

		except:
			
			return 'Unknown'		


def apply_offsets(rectangle, offsets):
	x, y, width, height = rectangle[3], rectangle[0], rectangle[1], rectangle[2]
	x_off, y_off = offsets
	return (x - x_off, x + width + x_off, y - y_off, y + height + y_off)

def preprocess_input(x, v2=True):
	x = x.astype('float32')
	x = x / 255.0
	if v2:
		x = x - 0.5
		x = x * 2.0
	return x