import os,sys
import numpy as np
from glob import glob

def variables_initializer():
	'''
	Initializes the required varibales in a function call

	Args:
	--------------------
	None

	Returns:
	------------------------------------------------------
	directory: path to data folder
	paths: paths to persons/classes in data
	face_names: list of labels of persons/classes
	imgs_per_folder: no of images to be taken from each class
	total_imgs = total_imgs in data

	'''
	directory = os.getcwd() +'/data/'
	paths = glob(directory+'*')
	face_names = [os.path.basename(i) for i in paths]

	if paths == []:
		sys.exit("Data Folder is empty.")

	minimum = 1e+8
	print("\nImage count summary:")
	print('Persons--Image count')

	for i in paths:
		temp = len(glob(f'{i}/*'))
		if temp < minimum:
			minimum = temp
		print(os.path.basename(i)+ '-' + str(temp))
	print('\n')

	imgs_per_folder = int(minimum)
	total_imgs = len(paths)*imgs_per_folder

	return directory,paths,face_names,imgs_per_folder,total_imgs

def store_references(paths,imgs_per_folder,emb_size,emb_model,data,face_names):
	'''
	Initializes the required varibales in a function call

	Args:
	--------------------------------------------------------
	paths: paths to persons/classes in data
	imgs_per_folder: no of images to be taken from each class
	emb_size: dimensions of space to project the face embeddings to
	emb_model:  trained embedding model
	data: array of faces sampling img_per_folders images per class
	face_names: list of labels of persons/classes

	Returns:
	----------------------------------------------
	Returns the reference embeddings of each class

	'''

	faces = np.zeros((len(paths),emb_size))
	pred = emb_model(data)

	# Creating embbeding references of each class
	for num_person in range(len(paths)):
		index_start = num_person * imgs_per_folder
		index_end = (num_person+1) * imgs_per_folder
		median_embedding = np.median(pred[ index_start : index_end],axis = 0)
		faces[num_person] = median_embedding

	# Saving the names of the persons to a text file (database) for quick loading the next time.
	np.save('./database/faces_emb_reference',faces)
	with open('./database/names_reference.txt','w') as face_txt:
		for name in face_names:
			face_txt.writelines(name + ' ')

	return faces