import cv2

class Frame_Handler:
	def __init__(self, source, fps=24):
		self.capture = cv2.VideoCapture(source)
		self.capture.set(cv2.CAP_PROP_FPS, 24)
		self.width  = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
		self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))  

	def get_frame(self):
		return self.capture.read()[1]

def change_color_dimension(frame):
	return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

def change_color_dimension_to_gray(frame):
	return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
