import os
from glob import glob
import numpy as np
import dlib
import cv2
from PIL import Image


def remove_undetected(directory ,detector ='hog'):

  '''
  Removes the undetected images in data

  Args:
  -----------------------------------------
  directory: path to the data folder
  detector: type of detector (Hog or Cnn) to detect faces

  Returns:
  ------------------------------------------
  Removes the undetected images in data
  Returns co-ordinates of rectangle bounding face in order (y1,y2,x1,x2)

  '''

  all_imgs = glob(f'{directory}*/*')

  for img in all_imgs:
    arr_img = np.asarray(Image.open(img))

    #Removes image if face could not be detected
    try:
      faces_detected = face_detector(arr_img,detector)
    except:
      print(img)
      os.remove(img)
      continue

    if (faces_detected == None) or (faces_detected == []):
      print(img)
      os.remove(img)

def face_detector(img,detector = 'hog'):

  '''
  Detects faces in images from data

  Args:
  -----------------------------------------
  img: numpy image array
  detector: type of detector (Hog or Cnn) to detect faces

  Returns:
  ------------------------------------------
  Returns co-ordinates of rectangle bounding face in order (y1,y2,x1,x2)

  '''

  if detector.lower() == 'hog':

    hogFaceDetector = dlib.get_frontal_face_detector()
    faceRects = hogFaceDetector(img, 1)
    faceRect = faceRects[0]
    if faceRect.top() > 0 and faceRect.bottom() > 0 and faceRect.left() > 0 and faceRect.right() > 0:
      return faceRect.top(), faceRect.bottom(), faceRect.left(), faceRect.right()
    else:
      return None

  elif detector.lower() == 'cnn':

    dnnFaceDetector = dlib.cnn_face_detection_model_v1('./database/dlib-models/mmod_human_face_detector.dat')
    rects = dnnFaceDetector(img, 1)
    faceRect  = rects[0]
    if faceRect.rect.top() > 0 and faceRect.rect.bottom() > 0 and faceRect.rect.left() > 0 and faceRect.rect.right() > 0:
      return faceRect.rect.top(),faceRect.rect.bottom(),faceRect.rect.left(),faceRect.rect.right()
    else:
      return None

def make_data_array(directory ,paths,img_size ,imgs_per_folder,total_imgs, check_detect,detector = 'hog'):

  '''
  Loads the data from disk to an array to speed up during training

  Args:
  -----------------------------------------
  directory: path to data folder
  paths: paths to persons/classes in data
  img_size = size of image to be resized to
  imgs_per_folder: no of images to be taken from each class
  total_imgs = total number of images in data folder
  detector: type of detector (Hog or Cnn) to detect faces
  check_detect: bool to check all imgs in data are detectable

  Returns:
  ------------------------------------------
  Returns the loaded input and its corresponding labels

  '''

  data = np.zeros((total_imgs,img_size,img_size,3),dtype = np.int)
  y = np.zeros((total_imgs))

  if check_detect:

    print("Removing undetected Images..")
    remove_undetected(directory,detector)

    # Re compute imgs per folder as value could be change by removed_undetected.
    minimum = 1e+8
    for i in paths:
      temp = len(glob(f'{i}/*'))
      if temp < minimum:
        minimum = temp
    imgs_per_folder = int(minimum)

    print("Removed undetected Images")
    print('-----------------------------------------\n')
  else:
    print("Skipping Detection Check")
    print('-----------------------------------------')
  print("Detecting Faces")

  # Storing imgs_per_folder faces from each class and its corresponding labels
  for index1,individual in enumerate(paths):
    for index2,picture in enumerate(glob(f'{individual}/*')[:imgs_per_folder]):
      img = np.asarray(Image.open(picture))

      y1,y2,x1,x2 = face_detector(img,detector)
      resized_img = cv2.resize(img[y1:y2,x1:x2],(img_size,img_size))

      data[index1*imgs_per_folder+index2] = resized_img
      y[index1*imgs_per_folder+index2] = index1

  print("Faces Detected and Loaded Successfully")
  return data,y,imgs_per_folder

