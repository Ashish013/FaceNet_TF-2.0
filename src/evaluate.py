from src.data_loader import face_detector
import numpy as np
import cv2
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt

def evaluate_on_img(img_size,face_names,faces,loaded_model,margin,detector):

	'''
	Evaluates/tests the emb_model on all the images in examples folder

	Arguments:
	-------------------------------------------
	img_size = size of image to be resized to
	face_names = list of names of classes/persons
	faces = embedding references of faces
	loaded_model = trained embedding model
	margin = intercluster distance
	detector = detector to detect faces in data

	Outputs:
	--------------------------------------------
	Annotated images with recognised faces

	'''

	for img_path in glob('./examples/*'):

		arr_pic = np.asarray(Image.open(img_path))
		y1,y2,x1,x2 = face_detector(arr_pic,detector)
		arr_pic_crop = cv2.resize(arr_pic[y1:y2,x1:x2],(img_size,img_size))
		original = loaded_model(np.expand_dims(arr_pic_crop,0))

		squared_dist = np.sum(np.square(original - faces),axis = 1)

		if np.min(squared_dist) < margin:
			detection = face_names[np.argmin(squared_dist)]
		else:
			detection = 'Not Found'

		rect_img = cv2.rectangle(arr_pic,(x1,y1),(x2,y2),(53,167,156),2)
		cv2.putText(rect_img,detection,(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,1.3,(255,255,255),2)
		plt.figure(figsize = (10,10))
		plt.imshow(rect_img)
		plt.title("Detected images in examples")
		plt.show()


def evaluate_on_webcam(img_size,face_names,faces,loaded_model,margin):

	'''
	Evaluates/tests the emb_model live on frames from web camera

	Arguments:
	-------------------------------------------
	img_size = size of image to be resized to
	face_names = list of names of classes/persons
	faces = embedding references of faces
	loaded_model = trained embedding model
	margin = intercluster distance
	detector = detector to detect faces in data

	Outputs:
	--------------------------------------------
	Live feed of annotated and recognised faces in frames

	'''

	cam = cv2.VideoCapture(0)
	cv2.namedWindow("Detector")

	while True:

	    ret, frame = cam.read()

	    if not ret:
	        print("Failed to grab frame")
	        break
	    try:
	    	detected = face_detector(frame,'hog')
	    except:
	    	detected = None

	    if (detected != []) and (detected != None):

	        y1,y2,x1,x2 = detected
	        frame_crop = cv2.resize(frame[y1:y2,x1:x2],(img_size,img_size))
	        original = loaded_model(np.expand_dims(frame_crop,0))
	        squared_dist = np.sum(np.square(original - faces),axis = 1)

	        if np.min(squared_dist) < margin:
	        	detection = face_names[np.argmin(squared_dist)]
	        else:
	            detection = 'Not Found'

	        rect_img = cv2.rectangle(frame,(x1,y1),(x2,y2),(53,167,156),2)
	        cv2.putText(rect_img,detection,(x2-110,y1-10),cv2.FONT_HERSHEY_SIMPLEX,1.3,(255,255,255),2)
	        cv2.imshow('Detector',rect_img)

	    else:
	        cv2.imshow('Detector',frame)

	    k = cv2.waitKey(1)

	    if k%256 == 27:
	        print("\nEscape hit !!\nClosing Window..")
	        break

	cam.release()
	cv2.destroyAllWindows()
