import argparse
import sys
import numpy as np
import matplotlib.pyplot as pylab
from decaf.scripts.jeffnet import JeffNet
from decaf.util import smalldata
import scipy.misc

import pygame.camera
import pygame.image
import pygame.surfarray

import threading


import nltk
from nltk.corpus import wordnet as wn



class ImageClassifier(threading.Thread):
  """ Class used for threading """
  
  def __init__(self, function_that_classifies):
    threading.Thread.__init__(self)
    self.runnable = function_that_classifies
    self.daemon = True

  def run(self):
    while True:
      self.runnable()




def classify_image():
  """ Function realizing the classification """
  if capturing:
    print "Classification (image: %d x %d)" % (img_numpy.shape[0], img_numpy.shape[1])
    scores = net.classify(img_numpy, center_only=True)
    top_detections = net.top_k_prediction(scores, len(scores))
    print top_detections[1][0:10]








# main function
data_root = '/home/rodner/data/deeplearning/models/'
requested_cam_size = (256,256)
enable_opengl = False

# deep net init
global net
net = JeffNet(data_root+'imagenet.jeffnet.epoch90', data_root+'imagenet.jeffnet.meta')

# pygame general initialization
pygame.init()

# camera initialization
pygame.camera.init()
camlist = pygame.camera.list_cameras()
if len(camlist) > 0:
  # setting up the camera
  print "-- List of cameras:"
  print camlist

  print "-- Selecting the first camera"
  cam = pygame.camera.Camera(camlist[0], requested_cam_size)
  cam.start()
  
  # get the camera size and setup up the window
  cam_size = cam.get_size()
  if enable_opengl:
    screen = pygame.display.set_mode( cam_size, (pygame.DOUBLEBUF | pygame.OPENGL | pygame.RESIZABLE)   )
  else:
    screen = pygame.display.set_mode( cam_size, (pygame.RESIZABLE)   )

  # starting the threading
  global img_numpy
  global capturing
  capturing = False

  thread = ImageClassifier(classify_image)
  thread.start()
  

  while True:
    # print "Capture image ..."
    img = cam.get_image()

    img_numpy = pygame.surfarray.array3d(img)
    capturing = True

    # test the conversion
    # pylab.imsave('test.png', img_numpy)

    screen.blit(img,(0,0))
    pygame.display.flip()
 

  #pygame.image.save(img, "photo.bmp")



  pygame.camera.quit()

else:
  print "No camera found!"
