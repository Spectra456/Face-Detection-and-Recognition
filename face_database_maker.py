from sklearn import neighbors
import face_recognition
import os
import pickle
import cv2
from face_detection import Face_Detection

#The training data would be all the face encodings from all the known images and the labels are their names
encodings = []
names = []

# Training directory
train_dir = os.listdir('train_dir/')
# Init face detector
yolo = Face_Detection(0.5,0.5, False)

# Loop through each person in the training directory
for person in train_dir:
    pix = os.listdir("train_dir/" + person)

    # Loop through each training image for the current person
    for person_img in pix:
        # Get the face encodings for the face in each image file
        print(person_img)
        face = cv2.imread("train_dir/" + person + "/" + person_img)
        face_bounding_boxes = detected = yolo.detect(face)
        print(face_bounding_boxes)

        #If training image contains none or more than faces, print an error message and exit
        if len(face_bounding_boxes) != 1:
            print(person + "/" + person_img + " contains none or more than one faces and can't be used for training.")
        else:
            try:
                face_enc = face_recognition.face_encodings(face,face_bounding_boxes)[0]
                # Add face encoding for current image with corresponding label (name) to the training data
                encodings.append(face_enc)
                names.append(person)
            except:
                os.remove("train_dir/" + person + "/" + person_img)

knn_clf = neighbors.KNeighborsClassifier(n_neighbors=2,leaf_size=128,algorithm='kd_tree', weights='distance', metric='euclidean')
knn_clf.fit(encodings,names)
                
facesDatabase = [encodings, names, knn_clf]

with open('facesDatabase.pickle', 'wb') as f:
     pickle.dump(facesDatabase, f)

print('Pickle file created')
            