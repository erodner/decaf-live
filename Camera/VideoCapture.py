"""Camera abstraction layer for Linux using mplayer and video decoding.

The Capture class provided from this module uses mplayer to decode a movie.

Author: Erik Rodner (adapted from an earlier version of Bjoern Barz)
"""

import logging
import cv2
import numpy as np
from PIL import Image
import VideoTools
from scipy.misc import imread, imresize
import glob

class Capture(object):
    """Provides access to video devices."""

    def __init__(self, mode, requested_cam_size=(640,480), url=None, videodir=None, videofile=None):
        
        self.requested_cam_size = requested_cam_size
        if mode=='download':
            if url:
                videofile, videolength = VideoTools.download_video( url )
                self.frames = VideoTools.decode_video(videofile, videolength)
            else:
                raise Exception("url unspecified")
        elif mode=='decode':
            if videofile:
                videolength = VideoTools.get_video_length(videofile)
                logging.info("The length of the video is: {0}".format(videolength))
                self.frames = VideoTools.decode_video(videofile, videolength)
            else:
                raise Exception("videofile unspecified")
        else: 
            if videodir:
                self.frames = glob.glob("%s/*.png" % (videodir))
            else:
                raise Exception("videodir unspecified")

        self.currentFrame = 0

    def __del__(self):
        pass

    def grabFrame(self):
        """Returns a snapshot from the device as PIL.Image.Image object."""
        
        data, w, h, orientation = self.grabRawFrame()

        return Image.fromstring("RGB", (w, h), data, "raw", "BGR", 0, orientation)


    def grabRawFrame(self):
        """Returns a snapshot from this device as raw pixel data.
        
        This function returns a 4-tuple consisting of the raw pixel data as string,
        the width and height of the snapshot and it's orientation, which is either
        1 (top-to-bottom) or -1 (bottom-to-top).
        """
        
        imgfn = self.frames[self.currentFrame]
        logging.info("Current frame: {0}".format(imgfn))
        cimg = imresize( imread(imgfn), size=( self.requested_cam_size[1], self.requested_cam_size[0] ) )
        height, width, depth = cimg.shape
        
        if self.currentFrame < len(self.frames)-1:
            self.currentFrame = self.currentFrame + 1
        
        # convert RGB to BGR :) 
        cimg = cimg[:,:,::-1]
        return cimg.tostring(), width, height, 1


    @staticmethod
    def enumerateDevices():
        devices = ()
        return devices
