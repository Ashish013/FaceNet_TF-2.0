import tensorflow as tf
from tensorflow.keras.layers import Dense,Input,Flatten
from tensorflow.keras import Model
import os,sys
import matplotlib.pyplot as plt
from glob import glob
from src.data_loader import make_data_array
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50

def emb_init(img_size,emb_size):
	'''
	Defines and returns the embedding model (ResNet50)

	Arguments:
	-------------------------------------------------
	img_size: size of image to be resized
	emb_size: dimensions of space to project the face embeddings to

	Outputs:
	-------------------------------------------------
	emb_model: initalized embedding model

	'''

	class ScaleLayer(tf.keras.layers.Layer):
		def __init__(self):
			super(ScaleLayer, self).__init__()

		def call(self, inputs):
			return tf.math.sqrt(tf.maximum(inputs,0.0), name ='Square_root')

	ptm = ResNet50(input_shape =(img_size,img_size,3), include_top=False,weights = None)
	x = Flatten()(ptm.output)
	x = Dense(emb_size,activation = 'softmax')(x)
	x = ScaleLayer()(x)
	emb_model = Model(ptm.input,x)

	return emb_model



def construct_database(directory,paths,emb_model,detector,check_detect):

	'''
	Constructs database from images in data folder provided trained emb_model weights are loaded and available

	Arguments:
	---------------------------------------------------
	directory: path to data folder
	paths: paths to classes of images in data folder
	emb_model:  trained emb_model
	check_detect: bool to check all imgs in data are detectable

	Outputs:
	----------------------------------------------------
	faces: numpy array of reference embeddings to classes/persons in data folder
	face_names: list of name of classes/persons in data folder

	'''

	face_names = [os.path.basename(i) for i in paths]
	emb_size = 128
	img_size = 200

	if paths == []:
		sys.exit("Data Folder is empty !")

	minimum = 1e+3

	print("\nImage count summary:")
	print('Persons | Image count\n')
	for i in paths:
		temp = len(glob(f'{i}/*'))
		if temp < minimum:
			minimum = temp
		print(os.path.basename(i) + ' | ' + str(temp))
	print('\n')

	imgs_per_folder = int(minimum)
	total_imgs = len(paths)*imgs_per_folder

	faces = np.zeros((len(paths),emb_size))
	data,y_names = make_data_array(directory,paths,img_size,imgs_per_folder,total_imgs,detector,check_detect)
	pred = emb_model(data)

	for num_person in range(len(paths)):
		index_start = num_person * imgs_per_folder
		index_end = (num_person+1) * imgs_per_folder
		median_embedding = np.median(pred[ index_start : index_end],axis = 0)
		faces[num_person] = median_embedding

	np.save('./database/faces_emb_reference',faces)
	with open('./database/names_reference.txt','w') as face_txt:
		for name in face_names:
			face_txt.writelines(name + ' ')

	pca_visualizer(plt,data,emb_model,y_names)

	return faces,face_names

