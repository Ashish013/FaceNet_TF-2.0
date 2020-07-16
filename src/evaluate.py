from src.data_loader import face_detector
import numpy as np
import cv2
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt

def evaluate_on_img(img_size,face_names,faces,loaded_model,margin,detector):

	'''
	Evaluates/tests the emb_model on all the images in examples folder

	Args:
	-------------------------------------------
	img_size = size of image to be resized to
	face_names: list of labels of persons/classes
	faces = embedding references of faces
	loaded_model = trained embedding model
	margin = intercluster distance
	detector: type of detector (Hog or Cnn) to detect faces

	Returns:
	--------------------------------------------
	Annotated images with recognised faces

	'''

	for img_path in glob('./examples/*'):

		arr_pic = np.asarray(Image.open(img_path))
		y1,y2,x1,x2 = face_detector(arr_pic,detector)
		arr_pic_crop = cv2.resize(arr_pic[y1:y2,x1:x2],(img_size,img_size))
		original = loaded_model(np.expand_dims(arr_pic_crop,0))

		squared_dist = np.sum(np.square(original - faces),axis = 1)

		# if distance between nearest cluster < margin then the identity is assigned to that of the cluster
		if np.min(squared_dist) < margin:
			detection = face_names[np.argmin(squared_dist)]
		else:
			detection = 'Not Found'

		face_area = x1 * (x2-x1)* y1*(y2-y1)

		# changes fontsize of text based on the rectangle bounding the face
		if face_area <= 1000000000:
			fontsize = 0.4
			thickness = 1
		elif face_area >= 15000000000:
			fontsize = 1.7
			thickness = 2
		else:
			fontsize = 1.3
			thickness = 2

		# Draws a rectangle around faces and annotates them
		cv2.rectangle(arr_pic,(x1,y1),(x2,y2),(53,167,156),2)
		(text_width, text_height) = cv2.getTextSize(detection,cv2.FONT_HERSHEY_SIMPLEX,fontScale = fontsize,thickness = thickness)[0]
		cv2.rectangle(arr_pic,(x1-4,y1+4),(x1+text_width+8,y1-text_height-8),(0,0,0),cv2.FILLED)
		cv2.putText(arr_pic,detection,(x1,y1),cv2.FONT_HERSHEY_SIMPLEX,fontsize,(255,255,255),thickness = thickness)
		plt.figure(figsize = (10,6))
		plt.imshow(arr_pic)
		plt.xticks([])
		plt.yticks([])
		plt.title("Detected images in examples")
		plt.show()


def evaluate_on_webcam(img_size,face_names,faces,loaded_model,margin):

	'''
	Evaluates/tests the emb_model live on frames from web camera

	Args:
	-------------------------------------------
	img_size = size of image to be resized to
	face_names: list of labels of persons/classes
	faces = embedding references of faces
	loaded_model = trained embedding model
	margin = intercluster distance

	Returns:
	--------------------------------------------
	Live feed of annotated and recognised faces in frames of video

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

			# if distance between nearest cluster < margin then the identity is assigned to that of the cluster
			if np.min(squared_dist) < margin:
				detection = face_names[np.argmin(squared_dist)]
			else:
				detection = 'Not Found'

			# Draws a rectangle around faces and annotates them
			fontsize = 1.1
			cv2.rectangle(frame,(x1,y1),(x2,y2),(53,167,156),2)
			(text_width, text_height) = cv2.getTextSize(detection,cv2.FONT_HERSHEY_SIMPLEX, fontsize,thickness = 2)[0]
			cv2.rectangle(frame,(x1-4,y1+4),(x1+text_width+8,y1-text_height-8),(0,0,0),cv2.FILLED)
			cv2.putText(frame,detection,(x1,y1),cv2.FONT_HERSHEY_SIMPLEX, fontsize,(255,255,255),2)
			cv2.imshow('Detector',frame)

		else:
			cv2.imshow('Detector',frame)

		k = cv2.waitKey(1)

		if k%256 == 27:
			print("\nEscape hit !!\nClosing Window..")
			break

	cam.release()
	cv2.destroyAllWindows()
