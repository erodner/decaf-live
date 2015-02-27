Decaf-Live 
==================

This is a real-time demonstration of a deep convolutional neural network trained on ImageNet. In particular, the DeCAF framework along with the pre-trained ImageNet model is used to perform classification of the complete image. The demo does not include object detection. Decaf-Live also supports offline video processing with direct youtube video download and video decoding.

Author: Erik Rodner (University of Jena, http://www.erodner.de)

1. Installation
---------------

The installation boils down to installing:
- DeCAF: https://github.com/UCB-ICSI-Vision-Group/decaf-release/
- Pre-trained ImageNet model: http://www.eecs.berkeley.edu/~jiayq/decaf_pretrained/
- pygame (for the GUI and camera support)
- optional: opencv for additional camera support
- optional: pafy python module for youtube video download
- optional: mplayer for video decoding 


2. Command line interface
--------------------------

Example (restricting recognition) to the artefact branch on ImageNet):

    python decaf-live.py -c artefact-categories.json

with the additional argument ``--downloadthumbs'', example images for ImageNet categories are downloaded, which are
shown during recognition.

Command line arguments:

    usage: decaf-live.py [-h] [-c CATEGORIES] [--width WIDTH] [--height HEIGHT]
                     [-m MODELDIR] [--thumbdir THUMBDIR] [--downloadthumbs]
                     [--threaded] [--nocenteronly]
                     [--offlinemode {download,decode,directory}] [--url URL]
                     [--videofile VIDEOFILE] [--videodir VIDEODIR]
                     [--loglevel {debug,info,warning,error,critical}]
                     [--delay DELAY] [--pooling {avg,none,max}]
                     [--poolingsize POOLINGSIZE]

    optional arguments:
    -h, --help            show this help message and exit
    -c CATEGORIES, --categories CATEGORIES
                        reduced list of categories as a JSON hash
    --width WIDTH         requested camera width
    --height HEIGHT       requested camera height
    -m MODELDIR, --modeldir MODELDIR
                        directory with model file and meta information
    --thumbdir THUMBDIR   directory with thumbnail images for the synsets
    --downloadthumbs      download non-existing thumbnail images
    --threaded            use classification thread
    --nocenteronly        disable center-only classification mode
    --offlinemode {download,decode,directory}
                        download|decode|directory
    --url URL             youtube video that will be downloaded in offline mode
    --videofile VIDEOFILE
                        video file that will be processed in offline mode
    --videodir VIDEODIR   directory with PNG files that will be processed in
                        offline mode
    --loglevel {debug,info,warning,error,critical}
                        log level
    --delay DELAY         delay (0=no delay, negative value=button wait,
                        positive value=milliseconds to wait)
    --pooling {avg,none,max}
                        type of pooling used
    --poolingsize POOLINGSIZE
                        pooling size

3. GUI usage
------------------------------------

The classification and image acquisition can be stopped by pressing space. The key ``q`` quits the program.

4. Acknowledgements
------------------------------------

A preliminary version of the camera module was implemented by Bjoern Barz.

