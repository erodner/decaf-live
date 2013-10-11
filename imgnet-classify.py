import argparse
import sys
import numpy as np
import matplotlib.pyplot as pylab
from decaf.scripts.jeffnet import JeffNet
from decaf.util import smalldata

data_root = '/home/dbv/deeplearning/models/'

if len(sys.argv)<2:\
  img = smalldata.lena()
else:
  imgfn = sys.argv[1]
  img = pylab.imread(imgfn)
  

net = JeffNet(data_root+'imagenet.jeffnet.epoch90', data_root+'imagenet.jeffnet.meta')

scores = net.classify(img)
print net.top_k_prediction(scores, 10)[1]
