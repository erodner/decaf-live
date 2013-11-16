import argparse

# main function
parser = argparse.ArgumentParser()
parser.add_argument( '-c', '--categories', help='reduced list of categories as a JSON hash', default=None )
parser.add_argument( '--width', type=int, help='requested camera width', default=256 )
parser.add_argument( '--height', type=int, help='requested camera height', default=256 )
parser.add_argument( '-m', '--modeldir', help='directory with model file and meta information', default='/home/rodner/data/deeplearning/models/' )
parser.add_argument( '--thumbdir', help='Directory with thumbnail images for the synsets', default='.' )
args = parser.parse_args()

import sys
import os
import math
import numpy as np
import matplotlib.pyplot as pylab
from decaf.scripts.jeffnet import JeffNet
from decaf.util import smalldata
import scipy.misc

import pygame.camera
import pygame.image
import pygame.surfarray

import threading

import json

from get_imagenet_thumbnails import get_imagenet_thumbnail








class ImageClassifier(threading.Thread):
  """ Class used for threading """
  
  def __init__(self, function_that_classifies):
    threading.Thread.__init__(self)
    self.runnable = function_that_classifies
    self.daemon = True

  def run(self):
    while True:
      self.runnable()



def create_thumbnail_cache(synsets, timgsize, thumbdir):
  """ load, rescale, and stote thumbnail images """

  maxk = 3
  maxtries = 10

  for synset in synsets:
    print "Caching thumbnails for synset %s" % (synset)
    tryk = 0
    successk = 0
    while tryk < maxtries and successk < maxk:
      thumbfn = thumbdir + os.path.sep + '%s_thumbnail_%04d.jpg' % ( synset, tryk )
      try:
        timgbig = pygame.image.load( thumbfn )
      except:
        tryk = tryk + 1
        continue
      
      print "Storing image %s %d: %s" % ( synset, successk, thumbfn )
      
      successk = successk + 1
      tryk = tryk + 1
      
      timg = pygame.transform.scale ( timgbig, timgsize )
      if not synset in thumbnail_cache:
        thumbnail_cache[synset] = []  
      thumbnail_cache[synset].append(timg)
  

  


def display_thumbnails(synsets, woffset, wsize):  

  screen.fill ( (0,0,0), pygame.Rect(woffset[0], woffset[1], wsize[0], wsize[1]) ) 
  maxk = 10
  maxtries = 10
  timgsize = ( wsize[0] / 3, wsize[1] / 3 )
  for i in range(len(synsets)):
    synset = synsets[i]
    for k in range( len(thumbnail_cache[synset]) ):
        x = timgsize[0] * k + woffset[0] 
        y = timgsize[1] * i + woffset[1]
        screen.blit(thumbnail_cache[synset][k],(x,y))


def display_results(synsets, scores, woffset, wsize):  
  # delete previous area
  screen.fill ( (0,0,0), pygame.Rect(woffset[0], woffset[1], wsize[0], wsize[1]) ) 

  myfont = pygame.font.SysFont("monospace", 15)
  rowsep = int ( wsize[1] / len(synsets) )
  rowoffset = rowsep/2

  sumscores = 0
  for i in range(len(synsets)):
    sumscores = sumscores + scores[i]

  for i in range(len(synsets)):
    text = '%s (%2f)' % (synsets[i], scores[i] / sumscores)
    #text = synsets[i]
    label = myfont.render(text, 1, (255,0,0), (0,0,0) )
    screen.blit(label, (woffset[0], woffset[1] + i * rowsep + rowoffset ))


def classify_image():
  """ Function realizing the classification """
  if capturing:
    camimg = pygame.surfarray.array3d(img)
    screen.blit(img,(img.get_width(),0))

    print "Classification (image: %d x %d)" % (camimg.shape[0], camimg.shape[1])
    print "\n"
    scores = net.classify(camimg, center_only=True)
    
    detections = net.top_k_prediction(scores, len(scores))
    # indices do not match with synset ids!
    print "ImageNet guesses (1000 categories): ", detections[1][0:5]
    print "\n"
    if categories:
      synindices = [ k for k in range(len(detections[1])) if detections[1][k] in categories ]
      descs = [ detections[1][k] for k in synindices ]
      synsets = [ categories[d] for d in descs ]
      scores_reduced = [ scores[detections[0][k]] for k in synindices ]
      print "Reduced set (%d categories): " % (len(categories)), descs[0:5]
      display_thumbnails( synsets, (0,camimg.shape[1]), (camimg.shape[0],camimg.shape[1]) )
      display_results ( descs[0:3], scores_reduced[0:3], (camimg.shape[0],camimg.shape[1]), (camimg.shape[0],camimg.shape[1]) )








global thumbnail_cache
thumbnail_cache = {}
global categories
categories = {}
if args.categories:
  categories = json.load( open( args.categories) )
  # preload synset thumbnails
  print "pre-downloading thumbnails ..."
  for synset in categories.keys():
    get_imagenet_thumbnail(synset, 6, verbose=True, overwrite=False, outputdir=args.thumbdir)
  create_thumbnail_cache ( categories.keys(), (100,80), args.thumbdir )

  # invert category map
  categories = dict( (v,k) for k, v in categories.items() )

data_root = args.modeldir
requested_cam_size = (args.width,args.height)

# OpenGL support not yet implemented
# gldrawPixels and the following command
# screen = pygame.display.set_mode( cam_size, (pygame.DOUBLEBUF | pygame.OPENGL | pygame.RESIZABLE)   )

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
  global screen
  screen = pygame.display.set_mode( ( 2*cam_size[0], 2*cam_size[1] ), (pygame.RESIZABLE)   )

  # starting the threading
  global img
  global capturing
  capturing = False

  thread = ImageClassifier(classify_image)
  thread.start()
  

  while True:
    # print "Capture image ..."
    img = cam.get_image()

    capturing = True

    # test the conversion
    # pylab.imsave('test.png', camimg)

    screen.blit(img,(0,0))
    pygame.display.flip()
 

  pygame.camera.quit()

else:
  print "No camera found!"
