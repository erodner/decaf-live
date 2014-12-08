Decaf-Live 
==================

This is a real-time demonstration of a deep convolutional neural network trained on ImageNet. In particular, the DeCAF framework along with the pre-trained ImageNet model is used to perform classification of the complete image. The demo does not include detection.

Author: Erik Rodner (University of Jena)


1. Installation
---------------

The installation boils down to installing:
- DeCAF: https://github.com/UCB-ICSI-Vision-Group/decaf-release/
- Pre-trained ImageNet model: http://www.eecs.berkeley.edu/~jiayq/decaf_pretrained/
- pygame (for camera support)


2. Usage
--------------

Example (restricting recognition) to the artefact branch on ImageNet):

    python decaf-live.py -c artefact-categories.json

with the additional argument ``--downloadthumbs'', example images for ImageNet categories are downloaded, which are
shown during recognition.

Command line arguments:
   usage: decaf-live.py [-h] [-c CATEGORIES] [--width WIDTH] [--height HEIGHT]
                     [-m MODELDIR] [--thumbdir THUMBDIR] [--downloadthumbs]
  optional arguments:
    -h, --help            show this help message and exit
    -c CATEGORIES, --categories CATEGORIES
                        reduced list of categories as a JSON hash
    --width WIDTH         requested camera width
    --height HEIGHT       requested camera height
    -m MODELDIR, --modeldir MODELDIR
                        directory with model file and meta information
    --thumbdir THUMBDIR   Directory with thumbnail images for the synsets
    --downloadthumbs      Download non-existing thumbnail images



