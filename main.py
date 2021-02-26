import tensorflow as tf
from glob import glob
import sys,os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense,Input,Flatten
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from argparse import ArgumentParser
from src.online_triplet_loss import Online_Mining
from src.embedding import emb_init,construct_database
from src.data_loader import make_data_array
from src.evaluate import evaluate_on_webcam,evaluate_on_img
from src.visualizer import img_plotter,pca_visualizer
from src.utils import variables_initializer,store_references


ap = ArgumentParser()
ap.add_argument('-t', '--train',required = True,type = str, help = 'Boolean which toggles model b/w training/evaluation mode')
ap.add_argument('-dw', '--download_weights', required = True,type = str, help = 'Boolean whether to download pretrained_weights')
ap.add_argument('-wp', '--weights_path', nargs = '?', default = './weights/trained/trained_weights', type = str, help = 'Path to weights in evaluate mode( Default : pretrained weights path)')
ap.add_argument('-m', '--margin', nargs = '?', default = 0.6 , type = float, help = 'Inter cluster distance / Min. margin between face embedding (Optimum Vaule: 0.4 - 0.6)')
ap.add_argument('-e' , '--epochs', nargs = '?', default = 20, type = int,help = 'Number of epochs to train')
ap.add_argument('-bs' , '--batch_size', type = int,help = 'batch_size (Note: It should be a factor of total images in data')
ap.add_argument('-lr', '--learning_rate',  nargs = '?', default = 1e-5, type = float, help = 'Learning rate ')
ap.add_argument('-d', '--detector',  nargs = '?', default = 'hog', type = str, help = "Args: Hog [Optimum] (or) Cnn [To be run on GPU]")
ap.add_argument('-chde', '--check_detect', nargs = '?', default = 'false', type = str, help = 'Boolean input to check whether all imgs in data are detectable' )
ap.add_argument('-wc', '--web_cam',nargs = '?', default = 'true', type = str, help = 'Toggle b/w webcam or Image display during evaluation mode' )
input_dict = vars(ap.parse_args())


if __name__ == "__main__":

	emb_size = 128
	img_size = 200
	margin = input_dict['margin']

	# Initalizes the embedding model
	emb_model = emb_init(img_size,emb_size)

	# Load pretrained weights to emb_model
	try:
		emb_model.load_weights(input_dict['weights_path'])
	except:
		if 'true' in input_dict['download_weights'].lower():

			from zipfile import ZipFile
			import gdown
			print("Downloading weights......")
			gdown.download('https://drive.google.com/uc?export=download&confirm=tOfl&id=1NYd6cQlewoQiFH71BHeOy2eTsZEvGzLg',output = './weights/pretrained.zip',quiet = False)

			with ZipFile("./weights/pretrained.zip", 'r') as zip:
				zip.extractall(path = './weights/')

			emb_model.load_weights('./weights/pretrained/ptm_weights')
			print('pretrained weights loaded successfully !!\n')

		elif 'false' in input_dict['download_weights'].lower():

			emb_model.load_weights('./weights/pretrained/ptm_weights')
			print('pretrained weights loaded successfully !!\n')

		else:
			print("pretrained weights not found !")
			sys.exit("Either download pretrained_weights manually or set the download_weights flag to True")


		# Converting str argument to boolean
		if 'true' in input_dict['check_detect'].lower():
			input_dict['check_detect'] = True

		elif 'false' in input_dict['check_detect'].lower():
			input_dict['check_detect'] = False

		else:
			sys.exit('check_detect is not a boolean !')
			

	if 'true' in (input_dict['train']).lower():

		# Toggles to training Mode

		# initializes required variables for training
		directory,paths,face_names,imgs_per_folder,total_imgs = variables_initializer()

		# Making sure batch size is divisible by total_imgs size
		try:
			if (imgs_per_folder*len(paths)) % input_dict['batch_size'] == 0:
				batch_size = input_dict['batch_size']
			else:
				sys.exit("Error : Try a different batch size argument as it should be a factor of length of all images in data !!")

		except:
			batch_size = (imgs_per_folder*len(paths))
			print("Batch size is taken as length of all images by default as batch_size argument was not provided......\n")

		#Loads the data from disk into an array
		data,y_names,imgs_per_folder = make_data_array(directory,paths,img_size,imgs_per_folder,total_imgs,input_dict['check_detect'],input_dict['detector'])

		img_plotter(plt,data)

		# Defining the model that trains the emb_model through online triplet loss using semi-hard/hard triplets
		input = Input(shape = (img_size,img_size,3))
		input1 = Input(shape = (1),batch_size = batch_size)
		x = emb_model(input)
		x = Online_Mining(margin)([x,input1])
		model = Model(inputs = [input,input1], outputs = x)

		generator = ImageDataGenerator(brightness_range = (0.4,1.8),horizontal_flip = True)
		train_generator = generator.flow(x = data, y = y_names, batch_size = batch_size,shuffle = True)

		def train_gen(batch_size=batch_size):
			while True:
				yield next(train_generator),np.zeros((batch_size))

		def custom_loss(y_true,y_pred):
			'''
			A dummy loss func to bypass an actual custon loss function
			Actual triplet loss is implemetned in online_triplet_loss
			'''
			return y_pred

		# Visualizes the emb_model output in 2 dimensions
		pca_visualizer(plt,data,emb_model,y_names,title = 'Embeddings (Before training)')

		epochs = input_dict['epochs']
		steps_per_epoch = int(total_imgs/batch_size) * 4
		model.compile(loss = custom_loss, optimizer= tf.keras.optimizers.Adam(input_dict['learning_rate']))
		hist = model.fit(train_gen(batch_size),steps_per_epoch = steps_per_epoch ,epochs= epochs)

		pca_visualizer(plt,data,emb_model,y_names,title = 'Embeddings (After training)')
		emb_model.save_weights('./weights/trained/trained_weights')

		faces = store_references(paths,imgs_per_folder,emb_size,emb_model,data,face_names)

	else:
		try:
			# Loading embedding reference of faces
			faces = np.load('./database/faces_emb_reference.npy')
			with open('./database/names_reference.txt','r') as face_txt:
				face_names = face_txt.readline().split()
		except:
			print('Faces_emb_reference or face_names file missing......\nTrying to construct database with images in data folder!')

			directory = os.getcwd() +'/data/'
			paths = glob(directory+'*')

			if paths != []:
				faces,face_names = construct_database(emb_size,img_size,emb_model,input_dict["detector"],input_dict["check_detect"])
			else:
				sys.exit("No Images in data to construct database\n")
				

	if 'true' in input_dict['web_cam'].lower():
		evaluate_on_webcam(img_size,face_names,faces,emb_model,margin)

	elif 'false' in input_dict['web_cam'].lower():
		evaluate_on_img(img_size,face_names,faces,emb_model,margin,input_dict['detector'])
	else:
		sys.exit("web_cam argument is not a boolean !")


		
