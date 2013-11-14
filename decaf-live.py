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


class ImageClassifier(threading.Thread):
  
  def __init__(self, function_that_classifies):
    threading.Thread.__init__(self)
    self.runnable = function_that_classifies
    self.daemon = True

  def run(self):
    while True:
      self.runnable()




def classify_image():
  if capturing:
    print "Running classifier"
    scores = net.classify(img_numpy)
    print net.top_k_prediction(scores, 10)[1]






#############

data_root = '/home/rodner/data/deeplearning/models/'

# deep net init
global net
net = JeffNet(data_root+'imagenet.jeffnet.epoch90', data_root+'imagenet.jeffnet.meta')

# camera init
pygame.camera.init()
camlist = pygame.camera.list_cameras()
# display init
pygame.init()
size=(640,480)
screen = pygame.display.set_mode(size)


if len(camlist) > 0:
  print "List of cameras:"
  print camlist

  cam = pygame.camera.Camera(pygame.camera.list_cameras()[0])
  cam.start()

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
