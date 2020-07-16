# FaceNet_TF-2.0
FaceNet_TF-2.0 is a naive implementation of [Face Net paper - 2015](https://arxiv.org/abs/1503.03832) by Florian Schroff, Dmitry Kalenichenko, James Philbin in Tensorflow 2.0. 
This implemenation opts online mode of semi - hard triplet mining over the all possible triplets approach which tends to learn better face embeddings.


![Demo working](/assets/Final-cut.gif)

### How does it work ?

The rough idea of the workflow can be summarized as:

- At first the faces are extracted from the training data and then a ResNet50 model is used to project these extracted faces onto a unit sphere in 128 dimensional euclidean 
space. 
- Next a triplet loss which mines for semi hard triplets is used to tweak the weights of the backbone ResNet50. The training goes on unitll a minimum intercluster distance 
is achieved between embeddings of each identy. 
- When a new image is evaluated on the model the identity is assigned to the closest cluster provided it lies in the margin.


### Flags:

Flags | Description
----------- | ------------------
'-t', '--train' | Boolean input which toggles model b/w train , evaluate mode (Required : True)
'-wc', '--web_cam' | Boolean input to toggle between web_cam /image input in evaluate mode (Required : True)
'-dw', '--download_weights' | Boolean input to download pretrained_weights or not
'-wp', '--weights_path' | Path to trained weights in evaluate mode (If not available; pretrained weights are used)
'-m', '--margin' | Inter cluster distance i.e, min. margin b/w 2 diff class face embeddings (Optimum Vaule: 0.4 - 0.6)
'-e' , '--epochs' | Number of epochs to train
'-bs' , '--batch_size' | batch_size ( Note: It should be a factor of total images in data)
'-lr', '--learning_rate' | Learning rate 
'-d', '--detector' | Hog (Optimum on CPU) or Cnn (Optimum on GPU)
'-chde', '--check_detect' | Boolean input to ensure all imgs in data are detectable


### How to run it ?
- First clone the repo to local machine; else [here](https://colab.research.google.com/drive/15lbTBNEZDsOdbIarumT5QQDdMWtx_96n?usp=sharing) is the link to run it on Google 
Colab 
- If cloned to local machine make sure **along with TF 2.0** all other dependencies in requirements.txt are installed.
- To train place the images structured into class name folders in to the data folder
- Next either pass True to '-dw' flag or manually download the pretrained weights from [here](https://drive.google.com/uc?export=download&confirm=tOfl&id=1NYd6cQlewoQiFH71BHeOy2eTsZEvGzLg) and place it in the weights folder
- Next change the working directory to current repo directory and run main.py with appropriate flags as in the table above

### Do I have to train it every time :question:
- Not at all once trained the script creates trained weights in weights folder; a txt, npy file in database folder. Make sure to make them available to script by not 
changing the file name or loaction. It is like a save feature of the model.

### Can I toggle to evaluate mode without training ?
- Absolutely but rember your trading accuracy here. To do so just place the images in data and make sure -dw flag is set or manually download pre trained weights and unzip in the weights folder.

