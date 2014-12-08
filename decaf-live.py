import argparse

# main function
parser = argparse.ArgumentParser()
parser.add_argument( '-c', '--categories', help='reduced list of categories as a JSON hash', default=None )
parser.add_argument( '--width', type=int, help='requested camera width', default=256 )
parser.add_argument( '--height', type=int, help='requested camera height', default=256 )
parser.add_argument( '-m', '--modeldir', help='directory with model file and meta information', default='/home/rodner/data/deeplearning/models/' )
parser.add_argument( '--thumbdir', help='directory with thumbnail images for the synsets', default='.' )
parser.add_argument( '--downloadthumbs', help='download non-existing thumbnail images', action='store_true')
parser.add_argument( '--threaded', help='use classification thread', action='store_true')
parser.add_argument( '--nocenteronly', help='disable center-only classification mode', action='store_true', default=False)
parser.add_argument( '--offlinemode', help='download|decode|directory', choices=['download', 'decode', 'directory'])
parser.add_argument( '--url', help='youtube video that will be downloaded in offline mode' )
parser.add_argument( '--videofile', help='video file that will be processed in offline mode' )
parser.add_argument( '--videodir', help='directory with PNG files that will be processed in offline mode' )
parser.add_argument( '--loglevel', help='log level', choices=['debug','info','warning','error','critical'], default='info')
parser.add_argument(' --delay', help='delay (0=no delay, negative value=button wait, positive value=milliseconds to wait)', type=int, default=0)
args = parser.parse_args()

import logging
numeric_level = getattr(logging, args.loglevel.upper(), None)
if not isinstance(numeric_level, int):
    raise ValueError('Invalid log level: %s' % args.loglevel)
logging.basicConfig(level=numeric_level)


import sys
import os
import math
import time
import numpy as np
import matplotlib.pyplot as pylab
from decaf.scripts.jeffnet import JeffNet
from decaf.util import smalldata
import scipy.misc
import functools

import pygame.image
import pygame.surfarray

import threading

import json

from get_imagenet_thumbnails import get_imagenet_thumbnail



class SingleFunctionThread(threading.Thread):
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

  logging.info("Loading thumbnails ...")
  for synset in synsets:
    logging.debug("Caching thumbnails for synset %s" % (synset))
    tryk = 0
    successk = 0
    while tryk < maxtries and successk < maxk:
      thumbfn = thumbdir + os.path.sep + '%s_thumbnail_%04d.jpg' % ( synset, tryk )
      try:
        timgbig = pygame.image.load( thumbfn )
      except:
        tryk = tryk + 1
        continue
      
      logging.debug("Storing image %s %d: %s" % ( synset, successk, thumbfn ))
      
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
    if synset in thumbnail_cache:
      for k in range( len(thumbnail_cache[synset]) ):
          x = timgsize[0] * k + woffset[0] 
          y = timgsize[1] * i + woffset[1]
          screen.blit(thumbnail_cache[synset][k],(x,y))


def display_results(synsets, scores, woffset, wsize):  
  # delete previous area
  screen.fill ( (0,0,0), pygame.Rect(woffset[0], woffset[1], wsize[0], wsize[1]) ) 

  myfont = pygame.font.SysFont("monospace", 15)
  myfont.set_bold(True)
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


def classify_image(center_only=True):
  """ Function realizing the classification """
  if capturing:
    screen.blit(img,(img.get_width(),0))
    
    camimg = np.transpose(pygame.surfarray.array3d(img), [1,0,2])

    # test the conversion
    #pylab.imsave('test.png', camimg)

    logging.info("Classification (image: %d x %d)" % (camimg.shape[1], camimg.shape[0]))
    scores = net.classify(camimg, center_only=center_only)
    
    detections = net.top_k_prediction(scores, len(scores))
    # indices do not match with synset ids!
    logging.info("ImageNet guesses (1000 categories): {0}".format(detections[1][0:5]))
    
    if categories:
      synindices = [ k for k in range(len(detections[1])) if detections[1][k] in categories ]
      descs = [ detections[1][k] for k in synindices ]
      synsets = [ categories[d] for d in descs ]
      scores_reduced = [ scores[detections[0][k]] for k in synindices ]
      logging.info("Reduced set ({0} categories): {1}".format(len(categories), descs[0:5]))
      display_thumbnails( synsets[0:3], (0,camimg.shape[0]), (camimg.shape[1],camimg.shape[0]) )
      display_results ( descs[0:3], scores_reduced[0:3], (camimg.shape[1],camimg.shape[0]), (camimg.shape[1],camimg.shape[0]) )









data_root = args.modeldir
requested_cam_size = (args.width,args.height)
enable_thumbnail_downloading = args.downloadthumbs

# OpenGL support not yet implemented
# gldrawPixels and the following command
# screen = pygame.display.set_mode( cam_size, (pygame.DOUBLEBUF | pygame.OPENGL | pygame.RESIZABLE)   )

# deep net init
global net
net = JeffNet(data_root+'imagenet.jeffnet.epoch90', data_root+'imagenet.jeffnet.meta')

# pygame general initialization
ret = pygame.init()
logging.debug("PyGame result: {0}".format(ret))
logging.debug("PyGame driver: {0}".format(pygame.display.get_driver()))


if args.offlinemode:
    from Camera.VideoCapture import Capture
    logging.info("Selecting the first camera")
    cam = Capture(requested_cam_size=requested_cam_size, url=args.url, videodir=args.videodir, mode=args.offlinemode, videofile=args.videofile)
else:
    from Camera.Capture import Capture
    logging.info("List of cameras:")
    logging.info(Capture.enumerateDevices())
    cam = Capture(index=0, requested_cam_size=requested_cam_size)


timg, width, height, orientation = cam.grabRawFrame()
cam_size = (width, height)
logging.info("Video camera size: {0}".format(cam_size))

# get the camera size and setup up the window
#cam_size = cam.get_size()


logging.debug("Initialize thumbnails")
global thumbnail_cache
thumbnail_cache = {}
global categories
categories = {}
if args.categories:
  categories = json.load( open( args.categories) )

# preload synset thumbnails
logging.debug("Pre-downloading thumbnails")
if enable_thumbnail_downloading:
  for idx, synset in enumerate(categories):
    logging.info("%d/%d %s" % ( idx, len(categories), synset))
    get_imagenet_thumbnail(synset, 6, verbose=True, overwrite=False, outputdir=args.thumbdir)
create_thumbnail_cache ( categories.keys(), (cam_size[0]/3, cam_size[1]/3), args.thumbdir )

# invert category map
categories = dict( (v,k) for k, v in categories.items() )



logging.debug("Initialize screen")
# open window
global screen
screen = pygame.display.set_mode( ( 2*cam_size[0], 2*cam_size[1] ), (pygame.RESIZABLE)   )

# starting the threading
global img
global capturing
capturing = True

if args.threaded:
  logging.debug("Initialize thread")
  thread = SingleFunctionThread(functools.partial(classify_image, not args.nocenteronly))
  thread.start()

while True:
  logging.debug("Capture image")
  capturing = False
  imgstring, w, h, orientation = cam.grabRawFrame()
  img = pygame.image.fromstring(imgstring[::-1], (w,h), "RGB" )
  img = pygame.transform.flip(img, True, True) 
  capturing = True

  if not args.threaded:
    classify_image(center_only=(not args.nocenteronly))

  screen.blit(img,(0,0))
  pygame.display.flip()
