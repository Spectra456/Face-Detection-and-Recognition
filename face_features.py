import pandas as pd
import pickle
import numpy as np
class Face_Features:
	def __init__(self, model_path='models/face_model.pkl'):
		with open(model_path, 'rb') as f:
			self.clf, self.labels = pickle.load(f, encoding='latin1')
			self.columns = ['Male', 'Asian', 'White', 'Black','Indian','0-5','10-15','20-35','45-65','65+','No Eyewear']

	def predict(self, faces_encodings):
		
		if faces_encodings != []:
			predict = pd.DataFrame(self.clf.predict_proba(faces_encodings), columns = self.labels)
			prediction = predict.loc[:, self.columns]
			return prediction
		else:
			return None

	def check_prediction(self, prediction):
		if prediction is not None:
			if round(prediction['Male'], 2) > 0.7:
				gender = 'M'
			else:
				gender = 'F'

			if round(prediction['No Eyewear'], 2) > 0.7:
				glass = 'No'
			else:
				glass = 'Yes'

			race = np.argmax(prediction[1:4])
			age = np.argmax(prediction[5:9])

			return age, race, gender, glass
		else:
			return None, None, None, None