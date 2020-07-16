import tensorflow as tf
from tensorflow.keras.layers import Dense,Input,Flatten
from tensorflow.keras import Model
import os,sys
import matplotlib.pyplot as plt
from glob import glob
from src.data_loader import make_data_array
from src.utils import variables_initializer,store_references
from src.visualizer import pca_visualizer
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50

def emb_init(img_size,emb_size):
	'''
	Defines and returns the embedding model (ResNet50)

	Args:
	-------------------------------------------------
	img_size: size of image to be resized
	emb_size: dimensions of space to project the face embeddings to

	Returns:
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



def construct_database(emb_size,img_size,emb_model,detector,check_detect):

	'''
	Constructs database from images in data folder provided trained emb_model weights are loaded and available

	Args:
	---------------------------------------------------
	emb_size: dimensions of space to project the face embeddings to
	img_size: size of image to be resized
	emb_model:  trained embedding model
	detector: type of detector (Hog or Cnn) to detect faces
	check_detect: bool to check all imgs in data are detectable

	Returns:
	----------------------------------------------------
	faces: numpy array of reference embeddings to classes/persons in data folder
	face_names: list of name of classes/persons in data folder

	'''

	directory,paths,face_names,imgs_per_folder,total_imgs = variables_initializer()
	data,y_names,imgs_per_folder = make_data_array(directory,paths,img_size,imgs_per_folder,total_imgs,check_detect,detector)
	faces = store_references(paths,imgs_per_folder,emb_size,emb_model,data,face_names)

	pca_visualizer(plt,data,emb_model,y_names,'Embeddings of data')

	return faces,face_names

