import face_recognition
import pickle
import numpy as np

class Face_Recognition:
	def __init__(self, model_path='models/facesDatabase.pickle'):
		self.confidence = 0.64
		with open(model_path, 'rb') as f:
			faceDatabase = pickle.load(f)
 
		self.encodings = faceDatabase[0]
		self.names = faceDatabase[1]
		self.knn_clf = faceDatabase[2]

		del faceDatabase

	def get_encodings(self, frame, face_locations):
		"""
		Get face encodings from dlib
		"""
		face_encodings = face_recognition.face_encodings(frame, face_locations)
		return face_encodings

	def predict(self, face_encodings):
		"""
		Comparing face with anothers in DB with KNN(Kdtree)
		"""
		name = self.knn_clf.predict([face_encodings])
		buffer = []
		for j in range(len(self.names)):
			if self.names[j] == name:
				# computing euclidean distance our encodings with all encodings of this face
				distance = np.linalg.norm([self.encodings[j]] - face_encodings, axis=1)
				# converting euclidean distance to probability
				buffer.append(1/(1  + distance))
		
		# finding average probability
		probability = np.mean(buffer)

		if probability >= self.confidence:
			return name[0]
		else:
		 	return 'Unknown'

