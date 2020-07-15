# FaceNet_TF-2.0
FaceNet_TF-2.0 is a naive implementation of [Face Net paper - 2015](https://arxiv.org/abs/1503.03832) by Florian Schroff, Dmitry Kalenichenko, James Philbin in Tensorflow 2.0. 
This implemenation opts online mode of semi - hard triplet mining over the all possible triplets approach which tends to learn better face embeddings.

### How does it work ?

The rough idea of the workflow can be summarized as:

- At first the faces are extracted from the training data and then a ResNet50 model is used to project these extracted faces onto a unit sphere in 128 dimensional euclidean 
space. 
- Next a triplet loss which mines for semi hard triplets is used to tweak the weights of the backbone ResNet50. The training goes on unitll a minimum intercluster distance 
is achieved between embeddings of each identy. 
- When a new image is evaluated on the model the identity is assigned to the closest cluster provided it lies in the margin.

### How to run it ??

