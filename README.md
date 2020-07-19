# FaceNet_TF-2.0
FaceNet_TF-2.0 is a naive implementation of [Face Net paper - 2015](https://arxiv.org/abs/1503.03832) by Florian Schroff, Dmitry Kalenichenko, James Philbin in Tensorflow 2.0. 
This implementation opts online mode of semi - hard triplet mining over the other possible triplets approach which tends to learn better face embeddings.

![Demo working](/assets/Final.gif)

### How does it work :grey_question:

The rough idea of the workflow can be summarized as:

- At first the faces are extracted from the training data using dlib module and then a ResNet50 model is used to project these extracted faces onto a unit sphere in 128 dimensional euclidean space. 
- Next a triplet loss which mines for semi hard triplets is used to tweak the weights of the backbone ResNet50. The training goes on untill a minimum intercluster distance 
is achieved between embeddings of each identy. 
- When a new image is evaluated on the model the identity is assigned to the closest cluster provided it lies in the margin.

### Architecture of the model:
![model_architecture](http://bamos.github.io/data/2016-01-19/optimization-after.png)

#### Image credits: [Brandon Amos](http://bamos.github.io/2016/01/19/openface-0.2.0/)

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


### How to run it :question:
- First clone the repo to your local machine; else [here](https://colab.research.google.com/drive/15lbTBNEZDsOdbIarumT5QQDdMWtx_96n?usp=sharing) is the link to run it on Google Colab 
- Once cloned to local machine make sure that **along with TF 2.0** all other dependencies in requirements.txt are installed.
- To train place the images structured into class names in the data folder (Around 15-20 images per identity hits the sweet spot)
- Next either pass True to '-dw' flag or manually download the pretrained weights from [here](https://drive.google.com/uc?export=download&confirm=tOfl&id=1NYd6cQlewoQiFH71BHeOy2eTsZEvGzLg) and place it in the weights folder
- Next change the working directory to current repository directory and run main.py with appropriate flags as in the table above
- And next based on the input to '-wc' flag the script toggles to evaluate mode. 
- If the '-wc' flag is set to **True** then model evaluates the model on frames from webcam's live feed.
- If set to **False** then the model evaluates all the images in the examples folder one by one as shown in the gif above.
- Oh and almost forgot hit **Esc** to quit the live feed in evaluate mode :sweat_smile:


### What are the "pretrained weights" you keep refering to :grey_question:
- These pretrained weights are not of any imagenet weights rather are weights obtained by training on personal images, a subset of celebrity images from this [kaggle dataset](https://www.kaggle.com/havingfun/100-bollywood-celebrity-faces?) for around 1000 epochs with an initial learning rate 5e+5


### Do I have to train it every time :question:
- Not at all once trained the script creates trained weights in weights folder; a txt, npy file in database folder. Make sure to make they are available to script by not 
changing the file names or directories.
- Let me rephrase it : If the script is like a game and you move or change the saved files; you start from Level 1 :grimacing::sweat_smile:


### Can I toggle to evaluate mode without training :grey_question:
- Absolutely :punch: but remember, here your trading accuracy for resources; as pretrained weights will be used for inference/evaluation.
- To do so just place the images in data and make sure '-dw' flag is set to True or manually download pre trained weights 
and unzip it in the weights folder.


### How to acheive better results :question:
- Make sure the data covers all the test cases you want your script to test on i.e, if you are wearing spectacles in all the images then most probably it will have a hard 
time recognising you without it. See (:nerd_face: -> :man:)
- Try to make sure your face is visible clear and big so as to facilitate a nice and tight crop of your face; else when resizing, noise is induced into the model 
which takes a toll on the predicted identity

**Please feel free to PR and let me know if there are any errors :relaxed:**
## Resources:

[Colab Link](https://colab.research.google.com/drive/15lbTBNEZDsOdbIarumT5QQDdMWtx_96n?usp=sharing)

[Pre Trained Weights](https://drive.google.com/uc?export=download&confirm=tOfl&id=1NYd6cQlewoQiFH71BHeOy2eTsZEvGzLg)

## References:

[Face Net paper](https://arxiv.org/abs/1503.03832)

[Triplet Loss and Online Triplet Mining in TensorFlow](https://omoindrot.github.io/triplet-loss)

[OpenFace 0.2.0: Higher accuracy and halved execution time](http://bamos.github.io/2016/01/19/openface-0.2.0/)
