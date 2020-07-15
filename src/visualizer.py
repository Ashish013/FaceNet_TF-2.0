import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def pca_visualizer(plt,data,emb_model,y_names):
	'''
	Displays the embeddings of data reduced to 2 dimensions
	'''
	pca_out = PCA(n_components=2).fit_transform(emb_model(data))
	plt.figure(figsize=(9, 4))
	plt.scatter(pca_out[:, 0], pca_out[:, 1], c = y_names)
	plt.title('PCA Visualizer')
	plt.show()

def img_plotter(plt,data):
	'''
	Displays a random image in data
	'''
	plt.imshow(data[np.random.choice(range(data.shape[0]))])
	plt.title('Random image in data')
	plt.show()

