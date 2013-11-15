import argparse
import sys
import numpy as np

try:
  import matplotlib.pyplot as pylab
except:
  import scipy.misc as pylab

from decaf.scripts.jeffnet import JeffNet
from decaf.util import smalldata

data_root = '/home/rodner/data/deeplearning/models/'

if len(sys.argv)<2:\
  img = smalldata.lena()
else:
  imgfn = sys.argv[1]
  img = pylab.imread(imgfn)
  

net = JeffNet(data_root+'imagenet.jeffnet.epoch90', data_root+'imagenet.jeffnet.meta')

print "Classify image"

for i in range(5):
  scores = net.classify(img)
  print "Okay"
  #print net.top_k_prediction(scores, 10)[1]
